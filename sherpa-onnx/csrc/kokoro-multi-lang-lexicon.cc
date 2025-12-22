// sherpa-onnx/csrc/kokoro-multi-lang-lexicon.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/kokoro-multi-lang-lexicon.h"

#include <fstream>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <strstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <mutex>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include <codecvt>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/phrase-matcher.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/tokenizer.h"

namespace sherpa_onnx {

class KokoroMultiLangLexicon::Impl {
 public:
  Impl(const std::string &g2p_model,
       const std::string &tokens, const std::string &lexicon,const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
    InitTokens(tokens);

    InitLexicon(lexicon);

    g2p_tokenizer_ = CreateTokenizer(g2p_model, token2id_);
  }

  template <typename Manager>
  Impl(Manager *mgr, const std::string &g2p_model,
       const std::string &tokens, const std::string &lexicon, const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
      : meta_data_(meta_data), debug_(debug) {
    InitTokens(mgr, tokens);

    InitLexicon(mgr, lexicon);

    // we assume you have copied data_dir from assets to some path

    g2p_tokenizer_ = CreateTokenizer(g2p_model, token2id_);
  }

  std::string DetectLanguage(const std::string &text, const std::string &default_voice) const {
    std::wstring ws = ToWideString(text);
  

    std::wregex we_zh(ToWideString("[\\u4e00-\\u9fff]+"));  // zh
    std::wregex we_jp(ToWideString("[\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FFF]+"));  // ja
    std::wregex we_kr(ToWideString("[\\uAC00-\\uD7A3\\u1100-\\u11FF]+"));  // ko
    std::wregex we_th(ToWideString("[\\u0E00-\\u0E7F]+"));
    std::wregex we_lo(ToWideString("[\\u0E80-\\u0EFF]+"));
    std::wregex we_km(ToWideString("[\\u1780-\\u17FF]+"));
    std::wregex we_my(ToWideString("[\\u1000-\\u109F]+"));
    std::wregex we_bo(ToWideString("[\\u0F00-\\u0FFF]+"));
    std::wregex we_sa(ToWideString("[\\u0900-\\u097F]+"));
    std::wregex we_ta(ToWideString("[\\u0B80-\\u0BFF]+"));
    std::wregex we_ml(ToWideString("[\\u0D00-\\u0D7F]+"));
    std::wregex we_te(ToWideString("[\\u0C00-\\u0C7F]+"));
    std::wregex we_kn(ToWideString("[\\u0C80-\\u0CFF]+"));
    std::wregex we_si(ToWideString("[\\u0D80-\\u0DFF]+"));
    std::wregex we_am(ToWideString("[\\u1200-\\u137F]+"));
    std::wregex we_ka(ToWideString("[\\u10A0-\\u10FF]+"));
  

    if (std::regex_search(ws, we_zh) && default_voice != "ja") {
      return "cmn";
    } else if (std::regex_search(ws, we_jp)) {
      return "ja";
    } else if (std::regex_search(ws, we_kr)) {
      return "ko";
    } else if (std::regex_search(ws, we_th)) {
      return "th";
    } else if (std::regex_search(ws, we_lo)) {
      return "lo";
    } else if (std::regex_search(ws, we_km)) {
      return "km";
    } else if (std::regex_search(ws, we_my)) {
      return "my";
    } else if (std::regex_search(ws, we_bo)) {
      return "bo";
    } else if (std::regex_search(ws, we_sa)) {
      return "sa";
    } else if (std::regex_search(ws, we_ta)) {
      return "ta";
    } else if (std::regex_search(ws, we_ml)) {
      return "ml";
    } else if (std::regex_search(ws, we_te)) {
      return "te";
    } else if (std::regex_search(ws, we_kn)) {
      return "kn";
    } else if (std::regex_search(ws, we_si)) {
      return "si";
    } else if (std::regex_search(ws, we_am)) {
      return "am";
    } else if (std::regex_search(ws, we_ka)) {
      return "ka";
    }
  
    return default_voice;
  }
  std::vector<TokenIDs> ConvertPhonemeToTokenIds(const std::string &text, const std::string &_lang) const {
    if (debug_) {
      SHERPA_ONNX_LOGE("ConvertPhonemeToTokenIds input text: '%s'", text.c_str());
    }

    std::vector<TokenIDs> ans;
    std::vector<int32_t> this_sentence;
    this_sentence.push_back(0);  // BOS token

    std::string current_text;
    bool in_bracket = false;
    std::vector<std::string> phonemes = SplitUtf8(text);
    int32_t max_len = meta_data_.max_token_len;

    if (debug_) {
      SHERPA_ONNX_LOGE("max_token_len: %d, phonemes size: %d", max_len, static_cast<int32_t>(phonemes.size()));
    }

    for (const auto &p : phonemes) {
      if (debug_) {
        SHERPA_ONNX_LOGE("Processing phoneme: '%s', current sentence size: %d", p.c_str(), static_cast<int32_t>(this_sentence.size()));
      }

      if (p == "[") {
        in_bracket = true;
        continue;
      } else if (p == "]") {
        in_bracket = false;
        if (!current_text.empty()) {
          // Check if the text is Chinese using regex
          std::wstring ws = ToWideString(current_text);
          std::wregex we_zh(ToWideString("[\\u4e00-\\u9fff]+"));
          std::wregex we_jp(ToWideString("[\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FFF]+"));
          std::wregex we_kr(ToWideString("[\\uAC00-\\uD7A3\\u1100-\\u11FF]+"));
          
          std::vector<std::vector<int32_t>> ids_vec;
          if (std::regex_match(ws, we_zh)) {
            if (debug_) {
              SHERPA_ONNX_LOGE("Converting CJK text: '%s'", current_text.c_str());
            }
            ids_vec = ConvertChineseToTokenIDs(current_text, _lang);
          } else {
            if (debug_) {
              SHERPA_ONNX_LOGE("Converting non-CJK text: '%s'", current_text.c_str());
            }
            ids_vec = ConvertEnglishToTokenIDs(current_text, _lang);
          }

          for (const auto &ids : ids_vec) {
            for (const auto id : ids) {
              if (this_sentence.size() + 2 > max_len) {  // +2 for EOS token
                this_sentence.push_back(0);  // EOS token
                ans.emplace_back(TokenIDs{std::move(this_sentence)});
                this_sentence.clear();
                this_sentence.push_back(0);  // BOS token for next sentence
              }
              if (debug_) {
                SHERPA_ONNX_LOGE("Adding token id: %d", id);
              }
              this_sentence.push_back(id);
            }
          }
          current_text.clear();
        }
        continue;
      }

      if (in_bracket) {
        current_text += p;
      } else if (token2id_.count(p)) {
        if (this_sentence.size() + 2 > max_len) {  // +2 for EOS token
          if (debug_) {
            SHERPA_ONNX_LOGE("Sentence reached max length, creating new sentence");
          }
          this_sentence.push_back(0);  // EOS token
          ans.emplace_back(TokenIDs{std::move(this_sentence)});
          this_sentence.clear();
          this_sentence.push_back(0);  // BOS token for next sentence
        }
        int32_t token_id = token2id_.at(p);
        if (debug_) {
          SHERPA_ONNX_LOGE("Converted phoneme '%s' to token id: %d", p.c_str(), token_id);
        }
        this_sentence.push_back(token_id);
      } else if (debug_) {
        SHERPA_ONNX_LOGE("Skip unknown phoneme: '%s'", p.c_str());
      }
    }

    if (this_sentence.size() > 1) {  // If we have more than just BOS token
      this_sentence.push_back(0);  // EOS token
      ans.emplace_back(TokenIDs{std::move(this_sentence)});
    }

    if (debug_) {
      for (const auto &v : ans) {
        std::ostringstream os;
        os << "\n";
        std::string sep;
        for (auto i : v.tokens) {
          os << sep << i;
          sep = " ";
        }
        os << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }
    }

    return ans;
  }

  std::vector<TokenIDs> ConvertTextToTokenIds(const std::string &_text, const std::string &_lang) const {
    std::string text = ToLowerCase(_text);
    if (debug_) {
      SHERPA_ONNX_LOGE("After converting to lowercase:\n%s", text.c_str());
    }

    std::vector<std::pair<std::string, std::string>> replace_str_pairs = {
        {"，", ","}, {":", ","},  {"、", ","}, {"；", ";"},   {"：", ":"},
        {"。", "."}, {"？", "?"}, {"！", "!"}, {"\\s+", " "},
    };
    for (const auto &p : replace_str_pairs) {
      std::regex re(p.first);
      text = std::regex_replace(text, re, p.second);
    }

    if (debug_) {
      SHERPA_ONNX_LOGE("After replacing punctuations and merging spaces:\n%s",
                       text.c_str());
    }

    // https://en.cppreference.com/w/cpp/regex
    // https://stackoverflow.com/questions/37989081/how-to-use-unicode-range-in-c-regex
    std::string expr_chinese = "([\\u4e00-\\u9fff]+)";
    std::string expr_japanese = "([\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FFF]+)";
    std::string expr_korean = "([\\uAC00-\\uD7A3\\u1100-\\u11FF]+)";
    std::string expr_not_cjk = "([^\\u4e00-\\u9fff\\u3040-\\u309F\\u30A0-\\u30FF\\uAC00-\\uD7A3\\u1100-\\u11FF\\[\\]]+)";
    std::string expr_bracket = "\\[(.*?)\\]";

    std::string expr_all = expr_bracket + "|" + expr_chinese + "|" + expr_japanese + "|" + expr_korean + "|" + expr_not_cjk;

    auto ws = ToWideString(text);
    std::wstring wexpr_all = ToWideString(expr_all);
    std::wregex we_all(wexpr_all);

    std::wstring wexpr_zh = ToWideString(expr_chinese);
    std::wregex we_zh(wexpr_zh);


    auto begin = std::wsregex_iterator(ws.begin(), ws.end(), we_all);
    auto end = std::wsregex_iterator();

    std::vector<TokenIDs> ans;

    for (std::wsregex_iterator i = begin; i != end; ++i) {
      std::wsmatch match = *i;
      std::wstring match_str = match.str();

      auto ms = ToString(match_str);
      uint8_t c = reinterpret_cast<const uint8_t *>(ms.data())[0];
      
      std::vector<std::vector<int32_t>> ids_vec;
      bool merge = false;
	    if (ms.size() >= 2 && ms[0] == '[' && ms.back() == ']') {
        std::string phoneme = ms.substr(1, ms.size() - 2);
        if (debug_) {
          SHERPA_ONNX_LOGE("Phoneme: %s", phoneme.c_str());
        }
        auto phoneme_ids = ConvertPhonemeToTokenIds(phoneme, _lang);
        for (const auto& token_ids : phoneme_ids) {
          std::vector<int32_t> converted_tokens;
          converted_tokens.reserve(token_ids.tokens.size());
          for (const auto& token : token_ids.tokens) {
            converted_tokens.push_back(static_cast<int32_t>(token));
          }
          ids_vec.push_back(std::move(converted_tokens));
        }
        merge = true;
      } else if (std::regex_match(match_str, we_zh)) {
        if (debug_) {
          SHERPA_ONNX_LOGE("Chinese: %s", ms.c_str());
        }
        ids_vec = ConvertChineseToTokenIDs(ms, _lang);
      } else {
        if (debug_) {
          SHERPA_ONNX_LOGE("Non-CJK: %s", ms.c_str());
        }
        ids_vec = ConvertEnglishToTokenIDs(ms, _lang);
      }

      for (const auto &ids : ids_vec) {
        if (ids.size() > 10 + 2 && !merge) {
          ans.emplace_back(ids);
        } else {
          if (ans.empty()) {
            ans.emplace_back(ids);
          } else {
            if(merge) {
              ans.back().tokens.back() = ids[1];
              ans.back().tokens.insert(ans.back().tokens.end(), ids.begin() + 2,
                                       ids.end());
            } else if ((ans.back().tokens.size() + ids.size() < 50) ||
                (ids.size() < 10)) {
              ans.back().tokens.back() = ids[1];
              ans.back().tokens.insert(ans.back().tokens.end(), ids.begin() + 2,
                                       ids.end());
            } else {
              ans.emplace_back(ids);
            }
          }
        }
      }
    }

    if (debug_) {
      for (const auto &v : ans) {
        std::ostringstream os;
        os << "\n";
        std::string sep;
        for (auto i : v.tokens) {
          os << sep << i;
          sep = " ";
        }
        os << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }
    }

    return ans;
  }

 private:
  bool IsPunctuation(const std::string &text) const {
    if (text == ";" || text == ":" || text == "," || text == "." ||
        text == "!" || text == "?" || text == "—" || text == "…" ||
        text == "\"" || text == "(" || text == ")" || text == "“" ||
        text == "”") {
      return true;
    }

    return false;
  }

  std::vector<int32_t> ConvertWordToIds(const std::string &w, const std::string &voice = "en-us") const {
    std::vector<int32_t> ans;
    if (word2ids_.count(w)) {
      ans = word2ids_.at(w);
      return ans;
    }

    std::wstring ws = ToWideString(w);
    std::wregex we_cjk(ToWideString("[\\u4e00-\\u9fff\\u3040-\\u309F\\u30A0-\\u30FF\\uAC00-\\uD7A3\\u1100-\\u11FF]+"));
    bool has_cjk = std::regex_search(ws, we_cjk);
    
    bool has_space = w.find(" ") != std::string::npos;
    
    if (has_cjk || has_space) {
      std::vector<std::string> words;
      
      if (has_space) {
        SplitStringToVector(w, " ", false, &words);
      } else {
        words.push_back(w);
      }
      
      for (const auto &word : words) {
        if (word2ids_.count(word)) {
          auto ids = word2ids_.at(word);
          ans.insert(ans.end(), ids.begin(), ids.end());
        }  else if (has_cjk) {
          std::vector<std::string> chars = SplitUtf8(word);
          for (const auto &c : chars) {
            if (word2ids_.count(c)) {
              auto ids = word2ids_.at(c);
              ans.insert(ans.end(), ids.begin(), ids.end());
            } else {
              if (debug_) {
                SHERPA_ONNX_LOGE("Use espeak-ng to handle the OOV CJK character: '%s'", c.c_str());
              }
              ProcessWithG2p(c, &ans, voice);
            }
          }
        } else {
          std::string detected_voice = DetectLanguage(word, voice);
          if (debug_) {
            SHERPA_ONNX_LOGE("Use espeak-ng to handle the OOV word: '%s'", word.c_str());
          }
          ProcessWithG2p(word, &ans, voice);
        }
      }
    } else {
      if (debug_) {
        SHERPA_ONNX_LOGE("Use espeak-ng to handle the OOV word: '%s'", w.c_str());
      }
      std::string detected_voice = DetectLanguage(w, voice);
      ProcessWithG2p(w, &ans, voice);
    }

    return ans;
  }
  void ProcessWithG2p(const std::string &text, std::vector<int32_t> *ans, const std::string &lang) const {
    if (debug_) {
      SHERPA_ONNX_LOGE("before process g2p word: '%s'", text.c_str());
    }
    std::vector<int64_t> ids = g2p_tokenizer_->Tokenize(text, lang);
    ans->insert(ans->end(), ids.begin(), ids.end());
    if (debug_) {
      std::unordered_map<int32_t, std::string> id2words_;
    
      for (const auto& pair : token2id_) {
          id2words_[pair.second] = pair.first;
      }
      
      SHERPA_ONNX_LOGE("after process g2p, result size: %zu", ans->size());
      for (size_t i = 0; i < ans->size(); ++i) {
          SHERPA_ONNX_LOGE("ans[%zu] = %d", i, (*ans)[i]);
          auto it = id2words_.find((*ans)[i]);
          if (it != id2words_.end()) {
              SHERPA_ONNX_LOGE("  word: %s", it->second.c_str());
          } else {
              SHERPA_ONNX_LOGE("  word: <unknown>");
          }
      }
    }
  }

  std::vector<std::vector<int32_t>> ConvertChineseToTokenIDs(
      const std::string &text, const std::string &voice = "en-us") const {
    std::vector<std::string> words = SplitUtf8(text);

    if (debug_) {
      std::ostringstream os;
      std::string sep = "";
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("after splitting into UTF8:\n%{public}s",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("after splitting into UTF8:\n%s", os.str().c_str());
#endif
    }

    std::vector<std::vector<int32_t>> ans;
    std::vector<int32_t> this_sentence;
    int32_t max_len = meta_data_.max_token_len;

    this_sentence.push_back(0);
	  PhraseMatcher matcher(&all_words_, words, debug_);
    for (const auto &w : matcher) {
      if(voice == "yue") {
        std::vector<int32_t> ids;
        ProcessWithG2p(w, &ids, voice);

        if (this_sentence.size() + ids.size() > max_len - 2) {
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }
        this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());
      } else {
        auto ids = ConvertWordToIds(w, voice);
        if (this_sentence.size() + ids.size() > max_len - 2) {
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }
        this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());
      }
    }

    if (this_sentence.size() > 1) {
      this_sentence.push_back(0);
      ans.push_back(std::move(this_sentence));
    }

    if (debug_) {
      for (const auto &v : ans) {
        std::ostringstream os;
        os << "\n";
        std::string sep;
        for (auto i : v) {
          os << sep << i;
          sep = " ";
        }
        os << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }
    }

    return ans;
  }

  std::vector<std::vector<int32_t>> ConvertEnglishToTokenIDs(
      const std::string &text, const std::string &voice) const {
    std::vector<std::string> words;
    bool non_space_lang = false;
    std::string effective_voice = voice.empty() ? meta_data_.voice : voice;

    if (!effective_voice.empty()) {
        // Check for prefixes indicating non-space-separating languages

        if (effective_voice == "cmn" || effective_voice == "yue" ||
          effective_voice == "ja") {
          non_space_lang = true;
        }
    }

    if (non_space_lang) {
        words = SplitUtf8(text);
    } else {
        SplitStringToVector(text, " ", false, &words);
        
        std::vector<std::string> processed_words;
        for (const auto &word : words) {
            std::string current_word;
            std::vector<std::string> chars = SplitUtf8(word);
            
            for (const auto &c : chars) {
                if (IsPunctuation(c)) {
                    if (!current_word.empty()) {
                        processed_words.push_back(current_word);
                        current_word.clear();
                    }
                    processed_words.push_back(c);
                } else {
                    current_word += c;
                }
            }
            
            if (!current_word.empty()) {
                processed_words.push_back(current_word);
            }
        }
        
        words = std::move(processed_words);
    }

    if (debug_) {
      std::ostringstream os;
      os << "After splitting to words and handling punctuation: ";
      std::string sep;
      for (const auto &w : words) {
        os << sep << w;
        sep = "_";
      }
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    std::vector<std::vector<int32_t>> ans;
    int32_t max_len = meta_data_.max_token_len;
    std::vector<int32_t> this_sentence;

    int32_t space_id = token2id_.at(" ");

    this_sentence.push_back(0);

    for (const auto &word : words) {
      if (IsPunctuation(word)) {
        this_sentence.push_back(token2id_.at(word));

        if (this_sentence.size() > max_len - 2) {
          // this sentence is too long, split it
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
          continue;
        }

        if (word == "." || word == "!" || word == "?" || word == ";") {
          // Note: You can add more punctuations here to split the text
          // into sentences. We just use four here: .!?;
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }
      } else if (word2ids_.count(word)) {
        const auto &ids = word2ids_.at(word);
        if (this_sentence.size() + ids.size() + 3 > max_len - 2) {
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }

        this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());
        this_sentence.push_back(space_id);
      } else {
        if (debug_) {
          SHERPA_ONNX_LOGE("Use espeak-ng to handle the OOV: '%s'",
                           word.c_str());
        }

        std::vector<int32_t> ids;
        ProcessWithG2p(word, &ids, effective_voice);

        if (this_sentence.size() + ids.size() + 3 > max_len - 2) {
          this_sentence.push_back(0);
          ans.push_back(std::move(this_sentence));

          this_sentence.push_back(0);
        }

        this_sentence.insert(this_sentence.end(), ids.begin(), ids.end());
        this_sentence.push_back(space_id);
      }
    }

    if (this_sentence.size() > 1) {
      this_sentence.push_back(0);
      ans.push_back(std::move(this_sentence));
    }

    if (debug_) {
      for (const auto &v : ans) {
        std::ostringstream os;
        os << "\n";
        std::string sep;
        for (auto i : v) {
          os << sep << i;
          sep = " ";
        }
        os << "\n";
        SHERPA_ONNX_LOGE("%s", os.str().c_str());
      }
    }

    return ans;
  }


  void InitTokens(const std::string &tokens) {
    std::ifstream is(tokens);
    InitTokens(is);
  }

  template <typename Manager>
  void InitTokens(Manager *mgr, const std::string &tokens) {
    auto buf = ReadFile(mgr, tokens);

    std::istrstream is(buf.data(), buf.size());
    InitTokens(is);
  }

  void InitTokens(std::istream &is) {
    token2id_ = ReadTokens(is);  // defined in ./symbol-table.cc
    token2id_["˥"] = 171; 
    token2id_["˧"] = 171; 
    token2id_["˨"] = 171;
    token2id_["˩"] = 171;
    token2id_["˧˥"] = 172;
    token2id_["˩˦"] = 172;
    token2id_["˦˥"] = 172;
    token2id_["˥˩"] = 169;
    token2id_["•"] = 173;
    token2id_["ɵ"] = 116;
    token2id_["ɭ"] = 54;
    token2id_["ɫ"] = 54;
    token2id_["ɝ"] = 85;
    token2id_["ʐ"] = 147;
    token2id_["õ"] = 57;
    token2id_.erase(":");

  }

  void InitLexicon(const std::string &lexicon) {
    if (lexicon.empty()) {
      return;
    }

    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      std::ifstream is(f);
      InitLexicon(is);
    }
  }

  template <typename Manager>
  void InitLexicon(Manager *mgr, const std::string &lexicon) {
    if (lexicon.empty()) {
      return;
    }

    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      auto buf = ReadFile(mgr, f);

      std::istrstream is(buf.data(), buf.size());
      InitLexicon(is);
    }
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

      if (word2ids_.count(word)) {
        num_warn += 1;
        if (num_warn < 10) {
          SHERPA_ONNX_LOGE("Duplicated word: %s at line %d:%s. Ignore it.",
                           word.c_str(), line_num, line.c_str());
        }
        continue;
      }

      while (iss >> token) {
        token_list.push_back(std::move(token));
      }

      std::vector<int32_t> ids = ConvertTokensToIds(token2id_, token_list);

      if (ids.empty() && word != "呣") {
        SHERPA_ONNX_LOGE(
            "Invalid pronunciation for word '%s' at line %d:%s. Ignore it",
            word.c_str(), line_num, line.c_str());
        continue;
      }

      word2ids_.insert({std::move(word), std::move(ids)});
    }

    for (const auto &[key, _] : word2ids_) {
      all_words_.insert(key);
    }
  }

 private:
  OfflineTtsKokoroModelMetaData meta_data_;

  // word to token IDs
  std::unordered_map<std::string, std::vector<int32_t>> word2ids_;
  std::unordered_set<std::string> all_words_;

  // tokens.txt is saved in token2id_
  std::unordered_map<std::string, int32_t> token2id_;
  std::unordered_map<int32_t, std::string> id2token_;

  std::unordered_map<char32_t, int32_t> phoneme2id_;

  bool debug_ = false;
  std::unique_ptr<Tokenizer> g2p_tokenizer_;
};

KokoroMultiLangLexicon::~KokoroMultiLangLexicon() = default;

KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    const std::string &g2p_model,
    const std::string &tokens, const std::string &lexicon,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
    : impl_(std::make_unique<Impl>(g2p_model, tokens, lexicon, meta_data, debug)) {}

template <typename Manager>
KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    Manager *mgr, const std::string &g2p_model, const std::string &tokens, const std::string &lexicon,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug)
    : impl_(std::make_unique<Impl>(mgr, g2p_model, tokens, lexicon, meta_data, debug)) {}

std::vector<TokenIDs> KokoroMultiLangLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string &lang) const {
  return impl_->ConvertTextToTokenIds(text, lang);
}

std::vector<TokenIDs> KokoroMultiLangLexicon::ConvertPhonemeToTokenIds(
  const std::string &text, const std::string &lang) const {
return impl_->ConvertPhonemeToTokenIds(text, lang);
}
#if __ANDROID_API__ >= 9
template KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    AAssetManager *mgr, const std::string &g2p_model, const std::string &tokens, const std::string &lexicon,
    const OfflineTtsKokoroModelMetaData &meta_data, bool debug);
#endif

#if __OHOS__
template KokoroMultiLangLexicon::KokoroMultiLangLexicon(
    NativeResourceManager *mgr, const std::string &g2p_model, const std::string &tokens,
    const std::string &lexicon, const OfflineTtsKokoroModelMetaData &meta_data,
    bool debug);
#endif

}  // namespace sherpa_onnx
