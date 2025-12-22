// sherpa-onnx/csrc/offline-tts-chatterbox-model.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

class OfflineTtsChatterboxModel {
 public:
  explicit OfflineTtsChatterboxModel(const OfflineTtsModelConfig &config);

  ~OfflineTtsChatterboxModel();

  Ort::Value Run(Ort::Value &tokens, const float *prompt, int64_t n_prompt,
               int64_t text_seed,   
               float exaggeration) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_MODEL_H_
