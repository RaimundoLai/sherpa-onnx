// Copyright (c) 2026 Xiaomi Corporation
//
// ONNX model-set primitives for the FasterLivePortrait pipeline.
//
// FasterLivePortrait is exported as several independent ONNX graphs.  This
// interface intentionally keeps the graphs independent and exposes their
// tensor contracts so callers can compose the pipeline without depending on
// OpenCV, Python, or a particular image codec.

#ifndef SHERPA_ONNX_CSRC_FASTER_LIVE_PORTRAIT_H_
#define SHERPA_ONNX_CSRC_FASTER_LIVE_PORTRAIT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

enum class FasterLivePortraitTensorType : int32_t {
  kFloat32 = 1,
  kFloat16 = 2,
  kInt64 = 3,
  kInt32 = 4,
  kUint8 = 5,
  kBool = 6,
};

struct FasterLivePortraitModelConfig {
  std::string name;
  std::string path;
};

struct FasterLivePortraitConfig {
  std::vector<FasterLivePortraitModelConfig> models;
  int32_t num_threads = 1;
  std::string provider = "cpu";
  bool debug = false;
};

struct FasterLivePortraitTensorInfo {
  std::string name;
  FasterLivePortraitTensorType type = FasterLivePortraitTensorType::kFloat32;
  std::vector<int64_t> shape;
};

struct FasterLivePortraitModelInfo {
  std::string name;
  std::string path;
  std::vector<FasterLivePortraitTensorInfo> inputs;
  std::vector<FasterLivePortraitTensorInfo> outputs;
};

// The data is borrowed only for the duration of Run().
struct FasterLivePortraitTensorInput {
  std::string name;
  FasterLivePortraitTensorType type =
      FasterLivePortraitTensorType::kFloat32;
  std::vector<int64_t> shape;
  const void *data = nullptr;
  size_t element_count = 0;
};

// The output owns its data and remains valid until the returned vector is
// destroyed by the caller.
struct FasterLivePortraitTensorOutput {
  std::string name;
  FasterLivePortraitTensorType type =
      FasterLivePortraitTensorType::kFloat32;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
};

class FasterLivePortraitModelSet {
 public:
  explicit FasterLivePortraitModelSet(const FasterLivePortraitConfig &config);
  ~FasterLivePortraitModelSet();

  const std::vector<FasterLivePortraitModelInfo> &GetModelInfos() const;

  std::vector<FasterLivePortraitTensorOutput> Run(
      const std::string &model_name,
      const std::vector<FasterLivePortraitTensorInput> &inputs) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FASTER_LIVE_PORTRAIT_H_
