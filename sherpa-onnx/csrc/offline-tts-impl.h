// sherpa-onnx/csrc/offline-tts-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts.h"

namespace sherpa_onnx {

class OfflineTtsImpl {
 public:
  virtual ~OfflineTtsImpl() = default;

  static std::unique_ptr<OfflineTtsImpl> Create(const OfflineTtsConfig &config);

  template <typename Manager>
  static std::unique_ptr<OfflineTtsImpl> Create(Manager *mgr,
                                                const OfflineTtsConfig &config);

  [[deprecated("Use Generate(text, GenerationConfig, callback) instead")]]
  virtual GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const {
    GenerationConfig config;
    config.sid = static_cast<int32_t>(sid);
    config.speed = speed;
    return Generate(text, config, std::move(callback));
  }

  virtual GeneratedAudio Generate(
      const std::string &text, int64_t sid, float speed, bool g2p, const std::string &lang,
      GeneratedAudioCallback callback = nullptr) const {
    GenerationConfig config;
    config.sid = static_cast<int32_t>(sid);
    config.speed = speed;
    if (g2p) {
      config.extra["g2p"] = "1";
    }
    if (!lang.empty()) {
      config.extra["lang"] = lang;
    }
    return Generate(text, config, std::move(callback));
  }

  virtual GeneratedAudio Generate(
      const std::string &text, const GenerationConfig &config,
      GeneratedAudioCallback callback = nullptr) const {
    SHERPA_ONNX_LOGE("Not implemented yet. Only some models support this");
    SHERPA_ONNX_LOGE("Please use sherpa-onnx > v1.12.30");
    return {};
  }

  virtual GeneratedAudio Generate(
      const std::string &text, const std::string &prompt_text,
      const std::vector<float> &prompt_samples, int32_t sample_rate,
      float speed = 1.0, int32_t num_step = 4,
      GeneratedAudioCallback callback = nullptr) const {
    SHERPA_ONNX_LOGE("Not implemented yet. Only some models support this");
    return {};
  }

  virtual GeneratedAudio Generate(
    const std::string &text, 
    const std::string &audio_dir,
    float speed = 1.0, 
    const std::string &lang = "en-us",
    float exaggeration = 0.5f,
    GeneratedAudioCallback callback = nullptr) const {
    throw std::runtime_error(
        "OfflineTtsImpl backend does not support zero-shot Generate()");
  }
  // Return the sample rate of the generated audio
  virtual int32_t SampleRate() const = 0;

  // Number of supported speakers.
  // If it supports only a single speaker, then it return 0 or 1.
  virtual int32_t NumSpeakers() const { return 1; }

  std::vector<int64_t> AddBlank(const std::vector<int64_t> &x,
                                int32_t blank_id = 0) const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_IMPL_H_
