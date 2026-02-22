// sherpa-onnx/csrc/offline-tts-miocodec-llama-model-config.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsMiocodecLlamaModelConfig {
  // Path to gpt_linacodec.onnx (KV-cache GPT language model)
  std::string model;

  // Path to campplus.onnx (CAM++ speaker embedding extractor)
  std::string campplus_model;

  // Path to miocodec_encoder.onnx
  std::string miocodec_encoder;

  // Path to miocodec_decoder.onnx
  std::string miocodec_decoder;

  // Path to embeddings.npz (tok_emb_weight, spk_proj_weight, spk_proj_bias)
  std::string embeddings;

  // Path to phoneme_vocab.json
  std::string tokens;

  // Lexicon file(s) for G2P. You can pass multiple files separated by ","
  std::string lexicon;

  // Path to G2P ONNX model (e.g. charsiug2p model, same as Kokoro)
  std::string g2p_model;

  // Autoregressive generation parameters
  float temperature = 0.7f;
  float top_p = 0.85f;
  int32_t max_tokens = 500;
  // Repetition penalty for audio token generation (1.0 = no penalty)
  float repetition_penalty = 1.0f;

  // Path to Perth watermarker ONNX model for audio watermarking (optional)
  std::string perth_watermarker;

  OfflineTtsMiocodecLlamaModelConfig() = default;

  OfflineTtsMiocodecLlamaModelConfig(
      const std::string &model, const std::string &campplus_model,
      const std::string &miocodec_encoder,
      const std::string &miocodec_decoder, const std::string &embeddings,
      const std::string &tokens, const std::string &lexicon,
      const std::string &g2p_model, float temperature, float top_p,
      int32_t max_tokens, float repetition_penalty,
      const std::string &perth_watermarker)
      : model(model),
        campplus_model(campplus_model),
        miocodec_encoder(miocodec_encoder),
        miocodec_decoder(miocodec_decoder),
        embeddings(embeddings),
        tokens(tokens),
        lexicon(lexicon),
        g2p_model(g2p_model),
        temperature(temperature),
        top_p(top_p),
        max_tokens(max_tokens),
        repetition_penalty(repetition_penalty),
        perth_watermarker(perth_watermarker) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_MODEL_CONFIG_H_
