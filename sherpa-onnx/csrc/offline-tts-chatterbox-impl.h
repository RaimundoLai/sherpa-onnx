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


#include "sherpa-onnx/csrc/offline-tts-chatterbox-model.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "tokenizers_cpp.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineTtsChatterboxImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsChatterboxImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsChatterboxModel>(config.model)) {
    auto blob =
        LoadBytesFromFile(config.model.chatterbox.tokenizer);
    tok_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  }

  int32_t SampleRate() const override { return 24000; } 

  int32_t NumSpeakers() const override {
    // chatterbox is a zero-shot model, so it does not have a fixed number of speakers.
    return -1;
  }


GeneratedAudio Generate(
    const std::string &text, 
    const std::string &audio_dir,
    float speed = 1.0, 
    const std::string &lang = "en-us",
    float exaggeration = 0.5f,
    GeneratedAudioCallback callback = nullptr) const override {

    std::vector<int32_t> ids = tok_->Encode(text);
    std::vector<int64_t> input_ids;
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
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHATTERBOX_IMPL_H_
