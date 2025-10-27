// sherpa-onnx/csrc/offline-tts-chatterbox-model-config.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsChatterboxModelConfig {
  std::string speech_encoder;
  std::string embed_tokens;
  std::string language_model;
  std::string conditional_decoder;
  std::string tokenizer;
  std::string lang;

  OfflineTtsChatterboxModelConfig() = default;

  OfflineTtsChatterboxModelConfig(
      const std::string &speech_encoder, const std::string &embed_tokens,
      const std::string &language_model,
      const std::string &conditional_decoder, const std::string &tokenizer,
      const std::string &lang)
      : speech_encoder(speech_encoder),
        embed_tokens(embed_tokens),
        language_model(language_model),
        conditional_decoder(conditional_decoder),
        tokenizer(tokenizer),
        lang(lang) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_MODEL_CONFIG_H_
