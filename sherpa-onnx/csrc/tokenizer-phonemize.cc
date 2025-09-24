#include "sherpa-onnx/csrc/tokenizer-phonemize.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <cctype>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/tokenizer.h"

namespace sherpa_onnx {

class ByT5TokenizerManual {
public:
    static const int32_t PAD_TOKEN_ID = 0;
    static const int32_t EOS_TOKEN_ID = 1;
    static const int32_t UNK_TOKEN_ID = 2;
private:
    std::unordered_map<int, int32_t> byte_to_token_id;
    std::unordered_map<int32_t, int> token_id_to_byte;
public:
    ByT5TokenizerManual() { initializeByteMappings(); }

    std::vector<int32_t> encode(const std::string& text, bool add_eos = true) const {
        std::vector<int32_t> token_ids;
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text.c_str());
        for (size_t i = 0; i < text.length(); ++i) {
            token_ids.push_back(byte_to_token_id.at(static_cast<int>(bytes[i])));
        }
        if (add_eos) { token_ids.push_back(EOS_TOKEN_ID); }
        return token_ids;
    }

    std::string decode(const std::vector<int32_t>& token_ids, bool skip_special_tokens = true) const {
        std::string result;
        for (int32_t token_id : token_ids) {
            if (skip_special_tokens &&
                (token_id == PAD_TOKEN_ID || token_id == EOS_TOKEN_ID || token_id == UNK_TOKEN_ID)) {
                continue;
            }

            if (token_id_to_byte.count(token_id)) {
                int byte_val = token_id_to_byte.at(token_id);
                result += static_cast<char>(byte_val);
            }
        }
        return result;
    }
private:
    void initializeByteMappings() {
        for (int byte_val = 0; byte_val < 256; ++byte_val) {
            int32_t token_id = byte_val + 3;
            byte_to_token_id[byte_val] = token_id;
            token_id_to_byte[token_id] = byte_val;
        }
    }
};

class TokenizerPhonemize : public Tokenizer {
public:
    TokenizerPhonemize(const std::string &model_path,
                       const std::unordered_map<std::string, int32_t>& token2id)
        : token2id_(token2id) {
        InitOrt(model_path);
        pad_token_id_ = byT5_tokenizer_.PAD_TOKEN_ID;
        eos_token_id_ = byT5_tokenizer_.EOS_TOKEN_ID;
    }

    std::vector<int64_t> Tokenize(const std::string &text,
                                  const std::string &lang) const override {
        
        std::string model_lang;
        auto it = espeak_to_model_lang_map.find(lang);
        if (it != espeak_to_model_lang_map.end()) {
            model_lang = it->second;
        } else {
            model_lang = lang;
        }

        
        std::string phoneme_string = G2pInfer(text, model_lang);

        
        std::vector<int64_t> final_token_ids;
        size_t i = 0;

        while (i < phoneme_string.length()) {
            unsigned char first_byte = phoneme_string[i];
            int char_len = 1;

            if (first_byte >= 0xC0 && first_byte < 0xE0) {
                char_len = 2;
            } else if (first_byte >= 0xE0 && first_byte < 0xF0) {
                char_len = 3;
            } else if (first_byte >= 0xF0 && first_byte < 0xF8) {
                char_len = 4;
            }
            if (i + char_len > phoneme_string.length()) {
                char_len = phoneme_string.length() - i;
            }

            if (char_len == 1 && std::isspace(first_byte)) {
                i += 1;
                continue;
            }

            std::string current_phoneme = phoneme_string.substr(i, char_len);

            auto map_entry = token2id_.find(current_phoneme);
            if (map_entry != token2id_.end()) {
                final_token_ids.push_back(map_entry->second);
            }  else {
                SHERPA_ONNX_LOGE("not found token: '%s' (hex: ", current_phoneme.c_str());
            }
    
            i += char_len;
        }
        return final_token_ids;
    }

    int32_t NumTokens() const {
        return max_token_id_ + 1;
    }

    const std::string &IdToToken(int32_t id) const {
        if (id < 0 || id > max_token_id_) {
            static const std::string invalid_token = "<INVALID>";
            return invalid_token;
        }
        return token_table_[id];
    }

private:
    std::string G2pInfer(const std::string& text, const std::string& lang) const {
        std::string full_text = "<" + lang + ">: " + text;
        std::vector<int32_t> encoded_ids = byT5_tokenizer_.encode(full_text);
        std::vector<int64_t> input_ids(encoded_ids.begin(), encoded_ids.end());
        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> decoder_input_ids = { (int64_t)pad_token_id_ };

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (int i = 0; i < 300; ++i) { // max_length=300
            std::vector<int64_t> input_shape = {1, (int64_t)input_ids.size()};
            std::vector<int64_t> attention_mask_shape = {1, (int64_t)attention_mask.size()};
            std::vector<int64_t> decoder_input_shape = {1, (int64_t)decoder_input_ids.size()};

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, attention_mask.data(), attention_mask.size(), attention_mask_shape.data(), attention_mask_shape.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, decoder_input_ids.data(), decoder_input_ids.size(), decoder_input_shape.data(), decoder_input_shape.size()));

            try {
                auto output_tensors = sess_->Run(Ort::RunOptions{nullptr},
                                                 session_input_names_char_.data(), input_tensors.data(), input_tensors.size(),
                                                 session_output_names_char_.data(), session_output_names_char_.size());

                const float* logits = output_tensors[0].GetTensorData<float>();
                auto& logits_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                int64_t vocab_size = logits_shape[2];
                const float* last_token_logits = logits + (logits_shape[1] - 1) * vocab_size;

                int64_t next_token_id = std::distance(last_token_logits,
                    std::max_element(last_token_logits, last_token_logits + vocab_size));

                if (next_token_id == eos_token_id_) {
                    break;
                }
                decoder_input_ids.push_back(next_token_id);
            } catch (const std::exception& e) {
                SHERPA_ONNX_LOGE("G2P Infer: ONNX inference error at step %d: %s", i, e.what());
                break;
            }
        }

        std::vector<int32_t> generated_ids;
        for (size_t j = 1; j < decoder_input_ids.size(); ++j) { // 跳過起始的 pad token
            generated_ids.push_back(static_cast<int32_t>(decoder_input_ids[j]));
        }

        return byT5_tokenizer_.decode(generated_ids);
    }

private:
    static const std::unordered_map<std::string, std::string> espeak_to_model_lang_map;

    void InitOrt(const std::string &model_path) {
        sess_opts_.SetIntraOpNumThreads(1);
        sess_opts_.SetInterOpNumThreads(1);
        
        try {
            auto buf = ReadFile(model_path);
            sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(), sess_opts_);
            GetInputNames(sess_.get(), &session_input_names_, &session_input_names_char_);
            GetOutputNames(sess_.get(), &session_output_names_, &session_output_names_char_);
        } catch (const std::exception& e) {
            SHERPA_ONNX_LOGE("TokenizerPhonemize: ONNX initialization failed: %s", e.what());
            throw;
        }
    }

private:
    std::unique_ptr<Ort::Session> sess_;
    Ort::Env env_{ORT_LOGGING_LEVEL_ERROR};
    Ort::SessionOptions sess_opts_;
    std::vector<std::string> session_input_names_;
    std::vector<const char *> session_input_names_char_;
    std::vector<std::string> session_output_names_;
    std::vector<const char *> session_output_names_char_;
    ByT5TokenizerManual byT5_tokenizer_;
    std::unordered_map<std::string, int32_t> token2id_;
    std::vector<std::string> token_table_;
    int32_t max_token_id_ = 0;
    int32_t pad_token_id_;
    int32_t eos_token_id_;
};

const std::unordered_map<std::string, std::string> TokenizerPhonemize::espeak_to_model_lang_map = {
    {"cmn", "zho-s"}, {"zho", "zho-s"}, {"yue", "yue"}, {"ja", "jpn"},
    {"ko", "kor"}, {"th", "tha"}, {"vi", "vie-n"}, {"vi-n", "vie-n"},
    {"vi-c", "vie-c"}, {"vi-s", "vie-s"}, {"hi", "hin"}, {"bn", "ben"},
    {"ta", "tam"}, {"id", "ind"}, {"tr", "tur"}, {"fa", "fas"},
    {"ar", "ara"}, {"kk", "kaz"}, {"km", "khm"}, {"my", "bur"},
    {"en", "eng-us"}, {"en-us", "eng-us"}, {"en-uk", "eng-uk"},
    {"en-gb", "eng-uk"}, {"fr", "fra"}, {"fr-ca", "fra-qu"}, {"de", "ger"},
    {"es", "spa"}, {"es-mx", "spa-me"}, {"it", "ita"}, {"pt", "por-po"},
    {"pt-pt", "por-po"}, {"pt-br", "por-bz"}, {"ru", "rus"}, {"nl", "dut"},
    {"pl", "pol"}, {"sv", "swe"}, {"da", "dan"}, {"no", "nob"},
    {"fi", "fin"}, {"el", "gre"}, {"el-grc", "grc"}, {"hu", "hun"},
    {"cs", "cze"}, {"ro", "ron"}, {"bg", "bul"}, {"uk", "ukr"},
    {"sr", "srp"}, {"hr", "hbs-latn"}, {"bs", "bos"}, {"sl", "slv"},
    {"sq", "sqi"}, {"is", "ice"}, {"ga", "gle"}, {"cy", "wel-nw"},
    {"af", "afr"}, {"sw", "swa"}
};

std::unique_ptr<Tokenizer> CreateTokenizer(
    const std::string &model_path, const std::unordered_map<std::string, int32_t>& token2id) {
    static std::mutex g2p_mutex;
    std::lock_guard<std::mutex> lock(g2p_mutex);
    return std::make_unique<TokenizerPhonemize>(model_path, token2id);
}

} // namespace sherpa_onnx