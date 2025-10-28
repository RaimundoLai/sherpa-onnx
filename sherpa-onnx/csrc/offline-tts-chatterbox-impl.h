// sherpa-onnx/csrc/offline-tts-chatterbox-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_IMPL_H_
#include "sherpa-onnx/csrc/wave-reader.h"

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

namespace sherpa_onnx {

class OfflineTtsChatterboxImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsChatterboxImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsChatterboxModel>(config.model)) {
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
    std::vector<std::string> chars = text_to_chars(text);
    PhraseMatcher matcher(&all_words_, chars, false);
    std::stringstream ss_segmented;
    for (const auto &w : matcher) {
        ss_segmented << w << " "; 
    }
    std::string segmented_text = ss_segmented.str();
    if (!segmented_text.empty()) {
        segmented_text.pop_back(); 
    }

    std::stringstream output_ss;
    std::u32string u32_segmented_text = utf8_to_u32(segmented_text);

    for (char32_t c : u32_segmented_text) {
        if (c >= 0x4e00 && c <= 0x9fff) {
            std::u32string glyph_u32(1, c);
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
            output_ss << u32_to_utf8(std::u32string(1, c));
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

std::string prepare_language(std::string txt, const std::string& lang) const {
    if (lang == "zh") {
        txt = chinese_cangjie_convert(txt);
    } else if (lang == "he") {
        txt = add_hebrew_diacritics(txt);
    } else if (lang == "ko") {
        txt = korean_normalize(txt);
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

    std::string processed_text = prepare_language(text, lang);
    SHERPA_ONNX_LOGE("processed_text: %s", processed_text.c_str());
    std::vector<int32_t> ids = tok_->Encode(processed_text);
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
    } else {
        SHERPA_ONNX_LOGE("No reference audio provided for zero-shot TTS.");
        exit(1);
    }

    Ort::Value audio = model_->Run(
        std::move(x_tensor),
        prompt_samples.data(),
        prompt_samples.size(),
        speed, // Note: speed is not used in Impl::Run, but text_seed is? Check parameters.
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
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsChatterboxModel> model_;
  std::unique_ptr<tokenizers::Tokenizer> tok_;
  std::unordered_map<std::u32string, std::string> word2cj_;
  std::unordered_map<std::string, std::vector<std::u32string>> cj2word_;
  std::unordered_set<std::string> all_words_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_IMPL_H_
