// sherpa-onnx/csrc/offline-tts-chatterbox-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-chatterbox-model.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <string.h>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

class RepetitionPenaltyLogitsProcessor {
 public:
  explicit RepetitionPenaltyLogitsProcessor(float penalty) : penalty_(penalty) {
    if (penalty <= 0) {
      SHERPA_ONNX_LOGE("Penalty must be a strictly positive float.");
    }
  }

  void apply(Ort::Value &logits,
             const std::vector<int64_t> &generated_tokens) {
    float *logits_data = logits.GetTensorMutableData<float>();
    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    int64_t vocab_size = logits_shape.back();

    for (int64_t token_id : generated_tokens) {
      if (token_id >= 0 && token_id < vocab_size) {
        float &score = logits_data[token_id];
        score = (score < 0) ? score * penalty_ : score / penalty_;
      }
    }
  }

 private:
  float penalty_;
};

}  // namespace

class OfflineTtsChatterboxModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING, "ChatterboxTTS-CPP"),
        sess_opts_(GetSessionOptions(config)) {
    auto &chatterbox = config_.chatterbox;
    Init(chatterbox.speech_encoder, chatterbox.embed_tokens,
         chatterbox.language_model, chatterbox.conditional_decoder);
  }

Ort::Value Run(Ort::Value &tokens, const float *prompt, int64_t n_prompt,
               int64_t text_seed,   
               float exaggeration = 0.5f) const {
    
    const int S3GEN_SR = 24000;
    const int64_t START_SPEECH_TOKEN = 6561;
    const int64_t STOP_SPEECH_TOKEN = 6562;
    const int MAX_NEW_TOKENS = 512;
    const float REPETITION_PENALTY = 1.2f;
    const int NUM_HIDDEN_LAYERS = 30;
    const int NUM_KEY_VALUE_HEADS = 16;
    const int HEAD_DIM = 64;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::RunOptions run_options{nullptr};
    
    std::vector<int64_t> generated_tokens = {START_SPEECH_TOKEN};
    RepetitionPenaltyLogitsProcessor penalty_processor(REPETITION_PENALTY);
    
    Ort::Value cond_emb{nullptr}, prompt_token{nullptr}, ref_x_vector{nullptr}, prompt_feat{nullptr};
    std::vector<Ort::Value> past_key_values;
    std::vector<int64_t> attention_mask_data;
    
    std::vector<float> combined_data;
    
    float exaggeration_val = exaggeration;

    for (int i = 0; i < MAX_NEW_TOKENS; ++i) {

        Ort::Value current_input_ids_tensor{nullptr};
        Ort::Value position_ids_tensor{nullptr};
        std::vector<int64_t> current_ids_data;
        std::vector<int64_t> current_pos_ids_data;
        std::vector<int64_t> current_shape;

        if (i == 0) {
            const int64_t* input_ids_data = tokens.GetTensorData<int64_t>();
            auto tokens_shape = tokens.GetTensorTypeAndShapeInfo().GetShape();
            current_ids_data.assign(input_ids_data, input_ids_data + tokens_shape[1]);
            current_shape = tokens_shape;

            current_pos_ids_data.resize(current_ids_data.size());
            for (size_t j = 0; j < current_ids_data.size(); ++j) {
                current_pos_ids_data[j] = (current_ids_data[j] >= START_SPEECH_TOKEN) ? 0 : (static_cast<int64_t>(j) - 1);
            }
        } else {
            current_ids_data.push_back(generated_tokens.back());
            current_shape = {1, 1};
            current_pos_ids_data.push_back(i);
        }

        current_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, current_ids_data.data(), current_ids_data.size(), current_shape.data(), current_shape.size());
        position_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, current_pos_ids_data.data(), current_pos_ids_data.size(), current_shape.data(), current_shape.size());

        std::vector<const char*> embed_input_names = {"input_ids", "position_ids", "exaggeration"};
        std::vector<Ort::Value> embed_inputs;
        embed_inputs.push_back(std::move(current_input_ids_tensor));
        embed_inputs.push_back(std::move(position_ids_tensor));
        embed_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, &exaggeration_val, 1, std::vector<int64_t>{1}.data(), 1));

        const char* embed_output_name = "inputs_embeds";
        auto embed_outputs = embed_tokens_session_->Run(run_options, embed_input_names.data(), embed_inputs.data(), embed_input_names.size(), &embed_output_name, 1);
        Ort::Value inputs_embeds = std::move(embed_outputs[0]);

        if (i == 0) {
            std::vector<int64_t> audio_shape = {1, n_prompt};
            Ort::Value audio_tensor = Ort::Value::CreateTensor(
                memory_info, const_cast<float *>(prompt), n_prompt,
                audio_shape.data(), audio_shape.size());

            std::vector<const char *> speech_encoder_output_names = {
                "audio_features", "audio_tokens", "speaker_embeddings", "speaker_features"};
            
            auto speech_encoder_outputs = speech_encoder_session_->Run(
                run_options, std::vector<const char*>{"audio_values"}.data(), // 注意：這裡的 input name 是 "audio"
                &audio_tensor, 1,
                speech_encoder_output_names.data(),
                speech_encoder_output_names.size());

            cond_emb = std::move(speech_encoder_outputs[0]);
            prompt_token = std::move(speech_encoder_outputs[1]);
            ref_x_vector = std::move(speech_encoder_outputs[2]);
            prompt_feat = std::move(speech_encoder_outputs[3]);

            auto cond_emb_shape_info = cond_emb.GetTensorTypeAndShapeInfo();
            auto inputs_embeds_shape_info = inputs_embeds.GetTensorTypeAndShapeInfo();
            auto cond_emb_shape = cond_emb_shape_info.GetShape();
            auto inputs_embeds_shape = inputs_embeds_shape_info.GetShape();
            std::vector<int64_t> combined_shape = {1, cond_emb_shape[1] + inputs_embeds_shape[1], cond_emb_shape[2]};

            combined_data.clear();
            combined_data.reserve(cond_emb_shape_info.GetElementCount() + inputs_embeds_shape_info.GetElementCount());
            combined_data.insert(combined_data.end(), cond_emb.GetTensorData<float>(), cond_emb.GetTensorData<float>() + cond_emb_shape_info.GetElementCount());
            combined_data.insert(combined_data.end(), inputs_embeds.GetTensorData<float>(), inputs_embeds.GetTensorData<float>() + inputs_embeds_shape_info.GetElementCount());
            inputs_embeds = Ort::Value::CreateTensor<float>(memory_info, combined_data.data(), combined_data.size(), combined_shape.data(), combined_shape.size());

            attention_mask_data.assign(combined_shape[1], 1);

            for (int j = 0; j < NUM_HIDDEN_LAYERS * 2; ++j) {
                std::vector<int64_t> past_shape = {1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM};
                past_key_values.push_back(Ort::Value::CreateTensor<float>(memory_info, nullptr, 0, past_shape.data(), past_shape.size()));
            }
        } else {
            attention_mask_data.push_back(1);
        }

        std::vector<std::string> llama_input_name_strings;
        llama_input_name_strings.push_back("inputs_embeds");
        llama_input_name_strings.push_back("attention_mask");
        for (int j = 0; j < NUM_HIDDEN_LAYERS; ++j) {
            llama_input_name_strings.push_back("past_key_values." + std::to_string(j) + ".key");
            llama_input_name_strings.push_back("past_key_values." + std::to_string(j) + ".value");
        }
        std::vector<const char*> llama_input_names;
        for (const auto& name : llama_input_name_strings) llama_input_names.push_back(name.c_str());

        std::vector<std::string> llama_output_name_strings;
        llama_output_name_strings.push_back("logits");
        for (int j = 0; j < NUM_HIDDEN_LAYERS; ++j) {
            llama_output_name_strings.push_back("present." + std::to_string(j) + ".key");
            llama_output_name_strings.push_back("present." + std::to_string(j) + ".value");
        }
        std::vector<const char*> llama_output_names;
        for (const auto& name : llama_output_name_strings) llama_output_names.push_back(name.c_str());
        
        std::vector<Ort::Value> llama_inputs;
        llama_inputs.push_back(std::move(inputs_embeds));
        std::vector<int64_t> attention_mask_shape = {1, static_cast<int64_t>(attention_mask_data.size())};
        llama_inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask_data.data(), attention_mask_data.size(), attention_mask_shape.data(), attention_mask_shape.size()));
        for (auto& val : past_key_values) llama_inputs.push_back(std::move(val));

        auto llama_outputs = llama_session_->Run(run_options, llama_input_names.data(), llama_inputs.data(), llama_input_names.size(), llama_output_names.data(), llama_output_names.size());

        Ort::Value logits = std::move(llama_outputs[0]);
        past_key_values.clear();
        for (size_t j = 1; j < llama_outputs.size(); ++j) past_key_values.push_back(std::move(llama_outputs[j]));

        auto logits_shape_info = logits.GetTensorTypeAndShapeInfo();
        auto logits_shape = logits_shape_info.GetShape();
        int64_t seq_len = logits_shape[1];
        int64_t vocab_size = logits_shape[2];
        float* last_token_logits_data = logits.GetTensorMutableData<float>() + (seq_len - 1) * vocab_size;

        std::vector<int64_t> next_token_logits_shape = {1, vocab_size};
        Ort::Value next_token_logits = Ort::Value::CreateTensor<float>(memory_info, last_token_logits_data, vocab_size, next_token_logits_shape.data(), next_token_logits_shape.size());

        penalty_processor.apply(next_token_logits, generated_tokens);

        const float* next_logits_ptr = next_token_logits.GetTensorData<float>();
        int64_t next_token = std::distance(next_logits_ptr, std::max_element(next_logits_ptr, next_logits_ptr + vocab_size));

        generated_tokens.push_back(next_token);

        if (next_token == STOP_SPEECH_TOKEN) {
            break;
        }
    }

    std::vector<int64_t> speech_tokens_data;
    if (generated_tokens.size() > 2) {
        speech_tokens_data.assign(generated_tokens.begin() + 1, generated_tokens.end() - 1);
    }

    auto prompt_token_count = prompt_token.GetTensorTypeAndShapeInfo().GetElementCount();
    const int64_t* prompt_token_data = prompt_token.GetTensorData<int64_t>();
    std::vector<int64_t> final_speech_tokens_data(prompt_token_data, prompt_token_data + prompt_token_count);
    final_speech_tokens_data.insert(final_speech_tokens_data.end(), speech_tokens_data.begin(), speech_tokens_data.end());

    std::vector<int64_t> speech_tokens_shape = {1, static_cast<int64_t>(final_speech_tokens_data.size())};
    Ort::Value speech_tokens = Ort::Value::CreateTensor<int64_t>(memory_info, final_speech_tokens_data.data(), final_speech_tokens_data.size(), speech_tokens_shape.data(), speech_tokens_shape.size());

    std::vector<const char *> decoder_input_names = {"speech_tokens", "speaker_embeddings", "speaker_features"};
    std::vector<Ort::Value> decoder_inputs;
    decoder_inputs.push_back(std::move(speech_tokens));
    decoder_inputs.push_back(std::move(ref_x_vector));
    decoder_inputs.push_back(std::move(prompt_feat));

    const char* decoder_output_name = "waveform";
    auto wav_outputs = cond_decoder_session_->Run(run_options, decoder_input_names.data(), decoder_inputs.data(), decoder_input_names.size(), &decoder_output_name, 1);
    
    return std::move(wav_outputs[0]);
}

 private:
  void Init(const std::string &speech_encoder,
            const std::string &embed_tokens, const std::string &language_model,
            const std::string &conditional_decoder) {
#ifdef _WIN32
    speech_encoder_session_ = std::make_unique<Ort::Session>(
        env_, StrToWstr(speech_encoder).c_str(), sess_opts_);
    embed_tokens_session_ = std::make_unique<Ort::Session>(
        env_, StrToWstr(embed_tokens).c_str(), sess_opts_);
    llama_session_ = std::make_unique<Ort::Session>(
        env_, StrToWstr(language_model).c_str(), sess_opts_);
    cond_decoder_session_ = std::make_unique<Ort::Session>(
        env_, StrToWstr(conditional_decoder).c_str(), sess_opts_);
#else
    speech_encoder_session_ = std::make_unique<Ort::Session>(
        env_, speech_encoder.c_str(), sess_opts_);
    embed_tokens_session_ =
        std::make_unique<Ort::Session>(env_, embed_tokens.c_str(), sess_opts_);
    llama_session_ = std::make_unique<Ort::Session>(env_, language_model.c_str(),
                                                    sess_opts_);
    cond_decoder_session_ = std::make_unique<Ort::Session>(
        env_, conditional_decoder.c_str(), sess_opts_);
#endif
  }

  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;

  std::unique_ptr<Ort::Session> speech_encoder_session_;
  std::unique_ptr<Ort::Session> embed_tokens_session_;
  std::unique_ptr<Ort::Session> llama_session_;
  std::unique_ptr<Ort::Session> cond_decoder_session_;
};

OfflineTtsChatterboxModel::OfflineTtsChatterboxModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineTtsChatterboxModel::~OfflineTtsChatterboxModel() = default;

Ort::Value OfflineTtsChatterboxModel::Run(Ort::Value &tokens, const float *prompt, int64_t n_prompt,
               int64_t text_seed,   
               float exaggeration) const {
  return impl_->Run(tokens, prompt, n_prompt, text_seed, exaggeration);
}

}  // namespace sherpa_onnx
