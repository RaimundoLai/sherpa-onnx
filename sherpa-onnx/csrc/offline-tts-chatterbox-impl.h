// sherpa-onnx/csrc/offline-tts-chatterbox-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_IMPL_H_
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/resample.h"

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>    
#include <unordered_map>
#include <locale>    
#include <codecvt> 
#include <algorithm>
#include <strstream>

#include "sherpa-onnx/csrc/offline-tts-chatterbox-model.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "tokenizers_cpp.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/phrase-matcher.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/file-utils.h"

// FST includes for text normalization
#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"
#include "kaldifst/csrc/text-normalizer.h"

namespace sherpa_onnx {

class OfflineTtsChatterboxImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsChatterboxImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsChatterboxModel>(config.model)),
        unknownIdsMapping_(CreateUnknownIdsMapping()) {
    auto blob =
        LoadBytesFromFile(config.model.chatterbox.tokenizer);
    tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
    if (!config.model.chatterbox.cangjie_dict.empty()) {
      LoadCangjieData(config.model.chatterbox.cangjie_dict);
    }
    if (!config.model.chatterbox.lexicon.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.model.chatterbox.lexicon, ",", false, &files);
      for (const auto &f : files) {
        auto buf = ReadFile(f);

        std::istrstream is(buf.data(), buf.size());
        InitLexicon(is);
      }
    }

    // Load FST text normalizers (for number/date conversion)
    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("chatterbox rule fst: %s", f.c_str());
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }

    if (!config.rule_fars.empty()) {
      if (config.model.debug) {
        SHERPA_ONNX_LOGE("Loading FST archives for Chatterbox");
      }
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);

      tn_list_.reserve(files.size() + tn_list_.size());

      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("chatterbox rule far: %s", f.c_str());
        }
        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(f));
        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));

          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }
      }

      if (config.model.debug) {
        SHERPA_ONNX_LOGE("FST archives loaded for Chatterbox!");
      }
    }
  }

  int32_t SampleRate() const override { return 24000; } 

  int32_t NumSpeakers() const override {
    // chatterbox is a zero-shot model, so it does not have a fixed number of speakers.
    return -1;
  }

std::u32string utf8_to_u32(const std::string& s) const {
    try {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        return conv.from_bytes(s);
    } catch (const std::exception& e) {
        SHERPA_ONNX_LOGE("UTF-8 to char32_t conversion failed: %s", e.what());
        return std::u32string();
    }
}

std::string u32_to_utf8(const std::u32string& s) const {
    try {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        return conv.to_bytes(s);
    } catch (const std::exception& e) {
        SHERPA_ONNX_LOGE("char32_t to UTF-8 conversion failed: %s", e.what());
        return std::string();
    }
}
std::vector<std::string> text_to_chars(const std::string &s) const {
    std::vector<std::string> chars;
    std::u32string u32_s = utf8_to_u32(s);
    for (char32_t c : u32_s) {
        chars.push_back(u32_to_utf8(std::u32string(1, c)));
    }
    return chars;
}
std::string _cangjie_encode(const std::u32string& glyph_u32) const {
    auto it = word2cj_.find(glyph_u32);
    if (it == word2cj_.end()) {
        return ""; 
    }

    const std::string& code = it->second;
    auto cj_it = cj2word_.find(code);

    if (cj_it == cj2word_.end() || cj_it->second.size() <= 1) {
        return code; 
    }

    const auto& words = cj_it->second;
    for (size_t i = 0; i < words.size(); ++i) {
        if (words[i] == glyph_u32) {
            if (i > 0) {
                return code + std::to_string(i); 
            } else {
                return code; 
            }
        }
    }
    return code; 
}
std::string korean_normalize(const std::string& text) const  {
    std::u32string u32_text = utf8_to_u32(text);
    std::u32string decomposed_text;
    decomposed_text.reserve(u32_text.length() * 2); 

    for (char32_t ch : u32_text) {
        if (ch >= 0xAC00 && ch <= 0xD7AF) { 
            char32_t base = ch - 0xAC00;
            char32_t initial = 0x1100 + base / (21 * 28);
            char32_t medial = 0x1161 + (base % (21 * 28)) / 28; 
            char32_t final_val = base % 28; 
            
            decomposed_text.push_back(initial);
            decomposed_text.push_back(medial);
            if (final_val > 0) {
                decomposed_text.push_back(0x11A7 + final_val);
            }
        } else {
            decomposed_text.push_back(ch);
        }
    }
    
    std::string result = u32_to_utf8(decomposed_text);
    size_t first = result.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first) {
        return "";
    }
    size_t last = result.find_last_not_of(" \t\n\r\f\v");
    return result.substr(first, (last - first + 1));
}


std::string chinese_cangjie_convert(const std::string& text) const  {
    if (all_words_.empty() || word2cj_.empty()) {
        if(all_words_.empty()) SHERPA_ONNX_LOGE("Chinese 'zh' processing skipped: Lexicon (all_words_) not loaded.");
        if(word2cj_.empty()) SHERPA_ONNX_LOGE("Chinese 'zh' processing skipped: Cangjie map (word2cj_) not loaded.");
        return text;
    }
    
    std::u32string u32_text = utf8_to_u32(text);
    std::stringstream output_ss;
    std::vector<std::string> chinese_buffer;
    
    for (char32_t c : u32_text) {
        if (c >= 0x4e00 && c <= 0x9fff) {
            chinese_buffer.push_back(u32_to_utf8(std::u32string(1, c)));
        } else {
            if (!chinese_buffer.empty()) {
                PhraseMatcher matcher(&all_words_, chinese_buffer, false);
                std::u32string chinese_segment;
                for (const auto &w : matcher) {
                    chinese_segment += utf8_to_u32(w);
                }
                
                for (char32_t ch : chinese_segment) {
                    if (ch >= 0x4e00 && ch <= 0x9fff) {
                        std::u32string glyph_u32(1, ch);
                        std::string cangjie_code = _cangjie_encode(glyph_u32);
                        
                        if (cangjie_code.empty()) {
                            output_ss << u32_to_utf8(glyph_u32);
                        } else {
                            for (char code_char : cangjie_code) {
                                output_ss << "[cj_" << code_char << "]";
                            }
                            output_ss << "[cj_.]";
                        }
                    } else {
                        output_ss << u32_to_utf8(std::u32string(1, ch));
                    }
                }
                chinese_buffer.clear();
            }
            
            output_ss << u32_to_utf8(std::u32string(1, c));
        }
    }
    
    if (!chinese_buffer.empty()) {
        PhraseMatcher matcher(&all_words_, chinese_buffer, false);
        std::u32string chinese_segment;
        for (const auto &w : matcher) {
            chinese_segment += utf8_to_u32(w);
        }
        
        for (char32_t ch : chinese_segment) {
            if (ch >= 0x4e00 && ch <= 0x9fff) {
                std::u32string glyph_u32(1, ch);
                std::string cangjie_code = _cangjie_encode(glyph_u32);
                
                if (cangjie_code.empty()) {
                    output_ss << u32_to_utf8(glyph_u32);
                } else {
                    for (char code_char : cangjie_code) {
                        output_ss << "[cj_" << code_char << "]";
                    }
                    output_ss << "[cj_.]";
                }
            } else {
                output_ss << u32_to_utf8(std::u32string(1, ch));
            }
        }
    }
    
    return output_ss.str();
}
std::string hiragana_normalize(const std::string& text) const  {
    SHERPA_ONNX_LOGE("Japanese 'ja' processing (pykakasi) is not implemented in C++ stub.");
    return text;
}

std::string add_hebrew_diacritics(const std::string& text) const  {
    SHERPA_ONNX_LOGE("Hebrew 'he' processing (dicta_onnx) is not implemented in C++ stub.");
    return text;
}

bool is_chinese(char32_t c) const {
    return (c >= 0x4E00 && c <= 0x9FFF) || // CJK Unified Ideographs
           (c >= 0x3400 && c <= 0x4DBF) || // CJK Unified Ideographs Extension A
           (c >= 0xF900 && c <= 0xFAFF) || // CJK Compatibility Ideographs
           (c >= 0x3000 && c <= 0x303F) || // CJK Symbols and Punctuation
           (c >= 0xFF00 && c <= 0xFFEF);   // Halfwidth and Fullwidth Forms
}

bool is_korean(char32_t c) const {
    return (c >= 0xAC00 && c <= 0xD7AF) || // Hangul Syllables
           (c >= 0x1100 && c <= 0x11FF) || // Hangul Jamo
           (c >= 0x3130 && c <= 0x318F) || // Hangul Compatibility Jamo
           (c >= 0xA960 && c <= 0xA97F) || // Hangul Jamo Extended-A
           (c >= 0xD7B0 && c <= 0xD7FF);   // Hangul Jamo Extended-B
}

// Supported language codes for Chatterbox TTS
// ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
bool is_supported_language(const std::string& tag) const {
    static const std::unordered_set<std::string> supported_langs = {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
        "sw", "tr", "zh"
    };
    return supported_langs.find(tag) != supported_langs.end();
}

// Check if text contains explicit language tags like [zh], [en], [ko], etc.
// Only matches supported language codes, ignores tags like [zh-tw]
bool has_language_tags(const std::string& txt) const {
    size_t pos = 0;
    while ((pos = txt.find('[', pos)) != std::string::npos) {
        size_t end_pos = txt.find(']', pos);
        if (end_pos != std::string::npos && end_pos > pos + 1) {
            std::string tag = txt.substr(pos + 1, end_pos - pos - 1);
            if (is_supported_language(tag)) {
                return true;
            }
        }
        pos++;
    }
    return false;
}

// Parse text with explicit language tags and process each segment
std::string parse_language_tags(const std::string& txt, const std::string& default_lang) const {
    std::string result;
    size_t pos = 0;
    std::string current_lang;
    
    while (pos < txt.length()) {
        // Look for next language tag
        size_t tag_start = txt.find('[', pos);
        
        if (tag_start == std::string::npos) {
            // No more tags, process remaining text with current language
            std::string remaining = txt.substr(pos);
            if (!remaining.empty()) {
                if (current_lang.empty()) {
                    // No language set yet, use default
                    current_lang = default_lang;
                }
                result += process_segment(remaining, current_lang);
            }
            break;
        }
        
        // Check if this is a valid language tag
        size_t tag_end = txt.find(']', tag_start);
        if (tag_end == std::string::npos) {
            // Malformed tag, treat rest as text
            std::string remaining = txt.substr(pos);
            if (!remaining.empty()) {
                if (current_lang.empty()) {
                    current_lang = default_lang;
                }
                result += process_segment(remaining, current_lang);
            }
            break;
        }
        
        std::string potential_tag = txt.substr(tag_start + 1, tag_end - tag_start - 1);
        
        // Check if it's a supported language tag
        bool is_valid_lang = is_supported_language(potential_tag);
        
        if (!is_valid_lang) {
            // Not a valid language tag, include it as text and continue searching
            // Process text up to this bracket first
            if (tag_start > pos) {
                std::string segment = txt.substr(pos, tag_start - pos);
                if (!segment.empty()) {
                    if (current_lang.empty()) {
                        current_lang = default_lang;
                    }
                    result += process_segment(segment, current_lang);
                }
            }
            // Include the bracket content as regular text
            result += txt.substr(tag_start, tag_end - tag_start + 1);
            pos = tag_end + 1;
            continue;
        }
        
        // Process text before this tag with current language
        if (tag_start > pos) {
            std::string segment = txt.substr(pos, tag_start - pos);
            if (!segment.empty()) {
                if (current_lang.empty()) {
                    // No language set yet, use default
                    current_lang = default_lang;
                }
                result += process_segment(segment, current_lang);
            }
        }
        
        // Update current language
        current_lang = potential_tag;
        pos = tag_end + 1;
    }
    
    return result;
}

// Process a text segment with the specified language
std::string process_segment(const std::string& segment, const std::string& lang) const {
    std::string processed = segment;
    
    if (lang == "zh") {
        processed = chinese_cangjie_convert(segment);
    } else if (lang == "ko") {
        processed = korean_normalize(segment);
    } else if (lang == "he") {
        processed = add_hebrew_diacritics(segment);
    }
    // For other languages like "en", no special processing needed
    
    return "[" + lang + "]" + processed;
}

std::string prepare_language(std::string txt, const std::string& lang) const {
    // Mode 2: Check if text contains explicit language tags
    if (has_language_tags(txt)) {
        return parse_language_tags(txt, lang);
    }
    
    // Mode 1: Auto-detect and tag languages (original behavior)
    if (lang == "zh" || lang == "ko") {
        std::u32string u32_txt = utf8_to_u32(txt);
        std::string result;
        std::u32string buffer;
        bool in_target_lang = false; 

        auto flush_buffer = [&](bool is_target) {
            if (buffer.empty()) return;
            std::string seg = u32_to_utf8(buffer);
            if (is_target) {
                if (lang == "zh") {
                    seg = chinese_cangjie_convert(seg);
                } else if (lang == "ko") {
                    seg = korean_normalize(seg);
                }
                result += "[" + lang + "]" + seg;
            } else {
                result += "[en]" + seg;
            }
            buffer.clear();
        };

        for (size_t i = 0; i < u32_txt.length(); ++i) {
            char32_t c = u32_txt[i];
            
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                buffer += c;
                continue;
            }

            bool is_char_target = (lang == "zh") ? is_chinese(c) : is_korean(c);

            if (buffer.empty()) {
                in_target_lang = is_char_target;
                buffer += c;
            } else {
                if (is_char_target != in_target_lang) {
                    flush_buffer(in_target_lang);
                    in_target_lang = is_char_target;
                }
                buffer += c;
            }
        }
        flush_buffer(in_target_lang);
        return result;
    } else if (lang == "he") {
        txt = add_hebrew_diacritics(txt);
    } 
    
    if (!lang.empty()) {
        txt = "[" + lang + "]" + txt;
    }
    return txt;
}
GeneratedAudio Generate(
    const std::string &text, 
    const std::string &audio_dir,
    float speed = 1.0, 
    const std::string &lang = "en",
    float exaggeration = 0.5f,
    GeneratedAudioCallback callback = nullptr) const override {

    std::string normalized_text = text;
    
    // Apply FST text normalization first (for number/date conversion)
    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        normalized_text = tn->Normalize(normalized_text);
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("After FST normalizing: %s", normalized_text.c_str());
        }
      }
    }

    std::string processed_text = prepare_language(normalized_text, lang);
    SHERPA_ONNX_LOGE("processed_text: %s", processed_text.c_str());
    std::vector<int32_t> ids = tok_->Encode(processed_text);
    for (int32_t &id : ids) {
      auto it = unknownIdsMapping_.find(id);
      if (it != unknownIdsMapping_.end()) {
        id = it->second;
      }
    }
    std::vector<int64_t> input_ids;
    input_ids.insert(input_ids.end(), {6563, 255}); 
    input_ids.insert(input_ids.end(), ids.begin(), ids.end());
    input_ids.insert(input_ids.end(), {0, 6561, 6561}); 
    
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> x_shape = {1,
                                      static_cast<int32_t>(input_ids.size())};
    Ort::Value x_tensor =
        Ort::Value::CreateTensor(memory_info, input_ids.data(),
                                 input_ids.size(), x_shape.data(), x_shape.size());

    std::vector<float> prompt_samples;
    int32_t prompt_sample_rate = -1;

   if (!audio_dir.empty()) {
        bool is_ok = false;
        prompt_samples = sherpa_onnx::ReadWave(
            audio_dir, 
            &prompt_sample_rate, 
            &is_ok
        );
        if (!is_ok) {
            SHERPA_ONNX_LOGE("Failed to read reference audio: %s", audio_dir.c_str());
            exit(1);
        }

        if (prompt_sample_rate != SampleRate()) {
          SHERPA_ONNX_LOGE(
              "Reference audio sample rate %d != target sample rate %d. "
              "Resampling...",
              prompt_sample_rate, SampleRate());

          float cutoff_hz =
              std::min(prompt_sample_rate, SampleRate()) / 2.0f * 0.95f;
          
          int32_t num_zeros = 8; 

          sherpa_onnx::LinearResample resampler(prompt_sample_rate, SampleRate(),
                                               cutoff_hz, num_zeros);

          std::vector<float> resampled_samples;
          resampler.Resample(prompt_samples.data(), prompt_samples.size(),
                             true,  // true = flush
                             &resampled_samples);

          prompt_samples = std::move(resampled_samples);
        }

    } else {
        SHERPA_ONNX_LOGE("No reference audio provided for zero-shot TTS.");
        exit(1);
    }

    Ort::Value audio = model_->Run(
        std::move(x_tensor),
        prompt_samples.data(),
        prompt_samples.size(),
        speed, 
        exaggeration
    );

    std::vector<int64_t> audio_shape =
        audio.GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    for (auto i : audio_shape) {
        total *= i;
    }

    const float *p = audio.GetTensorData<float>();
    if (p == nullptr) {
        SHERPA_ONNX_LOGE("Generate: GetTensorData<float>() returned nullptr!");
    }

    GeneratedAudio ans;
    ans.sample_rate = SampleRate();
    ans.samples = std::vector<float>(p, p + total);
    
    return ans;
}
  void InitLexicon(std::istream &is) {
    std::string word;
    std::vector<std::string> token_list;
    std::string token;

    std::string line;
    int32_t line_num = 0;
    int32_t num_warn = 0;
    while (std::getline(is, line)) {
      ++line_num;
      std::istringstream iss(line);

      token_list.clear();
      iss >> word;
      ToLowerCase(&word);
      all_words_.insert(word);
    }

  }
private:
  void LoadCangjieData(const std::string& file_path) {
      std::ifstream file(file_path);
      if (!file.is_open()) {
          SHERPA_ONNX_LOGE("CangjieConverter: Cannot open file: %s", file_path.c_str());
          return;
      }

      std::string line;
      while (std::getline(file, line)) {
          line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
          line.erase(line.find_last_not_of(" \t\n\r\f\v"));
          
          if (line.empty() || line == "[" || line == "]" || line == ",") {
              continue;
          }
          
          if (!line.empty() && line.back() == ',') {
              line.pop_back();
              line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
          }

          size_t start_quote = line.find('"');
          size_t end_quote = line.rfind('"');
          
          if (start_quote == std::string::npos || end_quote == std::string::npos || start_quote == end_quote) {
              continue;
          }

          std::string content = line.substr(start_quote + 1, end_quote - start_quote - 1);
          
          size_t tab_pos = content.find("\\t");
          if (tab_pos == std::string::npos) {
              continue;
          }

          std::string word_utf8 = content.substr(0, tab_pos);
          std::string code = content.substr(tab_pos + 2); 

          std::u32string word_u32 = utf8_to_u32(word_utf8);
          if (word_u32.empty() || code.empty()) {
              continue;
          }

          word2cj_[word_u32] = code;
          cj2word_[code].push_back(word_u32);
      }

      if(word2cj_.empty()) {
          SHERPA_ONNX_LOGE("Cangjie data %s is empty or parsing failed.", file_path.c_str());
      } else {
          SHERPA_ONNX_LOGE("Loaded Cangjie map with %d words.", (int)word2cj_.size());
      }
  }
 private:
  std::string LoadBytesFromFile(const std::string &path) const {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
      SHERPA_ONNX_LOGE("Cannot open %s", path.c_str());
      exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
  }

 private:
 static std::unordered_map<int32_t, int32_t> CreateUnknownIdsMapping() {
    std::unordered_map<int32_t, int32_t> mapping;
    mapping[2352] = 258;  // "€" -> "$"
    mapping[2353] = 1456; // "أ" -> "ا"
    mapping[2354] = 1456; // "إ" -> "ا"
    mapping[2355] = 1461; // "ئ" -> "ي"
    mapping[2356] = 1456; // "آ" -> "ا"
    mapping[2357] = 1459; // "ؤ" -> "و"
    mapping[2358] = 1490; // "ﻻ" -> "ل"
    mapping[2359] = 1456; // "ﺃ" -> "ا"
    mapping[2360] = 18;   // "ę" -> "e"
    mapping[2361] = 14;   // "ą" -> "a"
    mapping[2362] = 39;   // "ż" -> "z"
    mapping[2363] = 32;   // "ś" -> "s"
    mapping[2364] = 16;   // "ć" -> "c"
    mapping[2365] = 27;   // "ń" -> "n"
    mapping[2366] = 39;   // "ź" -> "z"
    mapping[2367] = 295;  // "Ś" -> "S"
    mapping[2368] = 302;  // "Ź" -> "Z"
    mapping[2369] = 302;  // "Ż" -> "Z"
    mapping[2370] = 279;  // "Ć" -> "C"
    mapping[2371] = 295;  // "Š" -> "S"
    mapping[2372] = 291;  // "Ő" -> "O"
    mapping[2373] = 1130; // "й" -> "и"
    mapping[2374] = 1127; // "ё" -> "е"
    mapping[2375] = 1093; // "Й" -> "И"
    mapping[2376] = 1082; // "Ё" -> "Е"
    mapping[2377] = 2159; // "が" -> "か"
    mapping[2378] = 2174; // "で" -> "て"
    mapping[2379] = 2166; // "じ" -> "し"
    mapping[2380] = 2170; // "だ" -> "た"
    mapping[2381] = 2175; // "ど" -> "と"
    mapping[2382] = 2181; // "ば" -> "は"
    mapping[2383] = 2163; // "げ" -> "け"
    mapping[2384] = 2164; // "ご" -> "こ"
    mapping[2385] = 2184; // "ぶ" -> "ふ"
    mapping[2386] = 2161; // "ぎ" -> "き"
    mapping[2387] = 7;    // "，" -> ","
    mapping[2388] = 5;    // "（" -> "("
    mapping[2389] = 11;   // "：" -> ":"
    mapping[2390] = 12;   // "；" -> ";"
    mapping[2391] = 13;   // "？" -> "?"
    mapping[2392] = 3;    // "！" -> "!"
    mapping[2393] = 257;  // "＃" -> "#"
    mapping[2394] = 6;    // " ）" -> ")"
    mapping[2395] = 1024; // "ά" -> "α"
    mapping[2396] = 1036; // "ό" -> "ο"
    mapping[2397] = 1006; // "ί" -> "ι"
    mapping[2398] = 1025; // "έ" -> "ε"
    mapping[2399] = 1026; // "ή" -> "η"
    mapping[2400] = 1027; // "ύ" -> "υ"
    mapping[2401] = 1045; // "ώ" -> "ω"
    mapping[2402] = 1000; // "Έ" -> "Ε"
    mapping[2403] = 1003; // "Ό" -> "Ο"
    mapping[2404] = 1001; // "Ή" -> "Η"
    mapping[2405] = 39;   // "ž" -> "z"
    mapping[2406] = 32;   // "š" -> "s"
    mapping[2407] = 34;   // "ū" -> "u"
    mapping[2408] = 32;   // "ş" -> "s"
    mapping[2409] = 291;  // "Ō" -> "O"
    mapping[2410] = 22;   // "ī" -> "i"
    mapping[2411] = 16;   // "č" -> "c"
    mapping[2412] = 31;   // "ř" -> "r"
    mapping[2413] = 14;   // "ă" -> "a"
    mapping[2414] = 1794; // "이" -> "ᄋ"
    mapping[2415] = 1783; // "기" -> "ᄀ"
    mapping[2416] = 1794; // "요" -> "ᄋ"
    mapping[2417] = 1794; // "에" -> "ᄋ"
    mapping[2418] = 1786; // "다" -> "ᄃ"
    mapping[2419] = 1794; // "을" -> "ᄋ"
    mapping[2420] = 1794; // "은" -> "ᄋ"
    mapping[2421] = 1792; // "서" -> "ᄉ"
    mapping[2422] = 1785; // "니" -> "ᄂ"
    mapping[2423] = 1794; // "어" -> "ᄋ"
    mapping[2424] = 18;   // "ě" -> "e"
    mapping[2425] = 34;   // "ů" -> "u"
    mapping[2426] = 279;  // "Č" -> "C"
    mapping[2427] = 27;   // "ň" -> "n"
    mapping[2428] = 17;   // "ď" -> "d"
    mapping[2429] = 33;   // "ť" -> "t"
    mapping[2430] = 15;   // "♭" -> "b"
    mapping[2431] = 25;   // "ľ" -> "l"
    mapping[2432] = 25;   // "ĺ" -> "l"
    mapping[2433] = 20;   // "ğ" -> "g"
    mapping[2434] = 285;  // "İ" -> "I"
    mapping[2435] = 295;  // "Ş" -> "S"
    mapping[2436] = 1699; // "ड़" -> "ड"
    mapping[2437] = 1700; // "ढ़" -> "ढ"
    mapping[2438] = 1694; // "ज़" -> "ज"
    mapping[2439] = 1709; // "फ़" -> "फ"
    mapping[2440] = 1688; // "ख़" -> "ख"
    mapping[2441] = 1687; // "क़" -> "क"
    mapping[2442] = 1689; // "ग़" -> "ग"
    mapping[2443] = 999;  // "Ά" -> "Α"
    mapping[2444] = 1006; // "ϊ" -> "ι"
    mapping[2445] = 1002; // "Ί" -> "Ι"
    mapping[2446] = 1004; // "Ύ" -> "Υ"
    mapping[2447] = 1005; // "Ώ" -> "Ω"
    mapping[2448] = 1006; // "ΐ" -> "ι"
    mapping[2449] = 1027; // "ϋ" -> "υ"
    mapping[2450] = 34;   // "ũ" -> "u"
    mapping[2451] = 34;   // "ụ" -> "u"
    mapping[2452] = 28;   // "ọ" -> "o"
    mapping[2453] = 14;   // "ạ" -> "a"

    return mapping;
  }
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsChatterboxModel> model_;
  std::unique_ptr<tokenizers::Tokenizer> tok_;
  std::unordered_map<std::u32string, std::string> word2cj_;
  std::unordered_map<std::string, std::vector<std::u32string>> cj2word_;
  std::unordered_set<std::string> all_words_;
  const std::unordered_map<int32_t, int32_t> unknownIdsMapping_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_IMPL_H_