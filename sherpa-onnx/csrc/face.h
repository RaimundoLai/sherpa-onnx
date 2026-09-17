//
// Copyright (c)  2026  Xiaomi Corporation
//
// Face detection and face embedding primitives.  The image
// interface deliberately operates on packed raw pixels so that the core does
// not need to depend on an image codec or OpenCV.

#ifndef SHERPA_ONNX_CSRC_FACE_H_
#define SHERPA_ONNX_CSRC_FACE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

enum class ImageFormat : int32_t {
  kRGB = 0,
  kBGR = 1,
  kRGBA = 2,
  kBGRA = 3,
};

struct ImageView {
  const uint8_t *data = nullptr;
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  int32_t stride = 0;
  ImageFormat format = ImageFormat::kRGB;

  bool IsValid() const;
  int32_t EffectiveStride() const;
};

struct FaceDetection {
  float score = 0;
  // x1, y1, x2, y2 in source-image pixels.
  float bbox[4] = {0, 0, 0, 0};
  // right eye, left eye, nose, right mouth, left mouth; x/y pairs.
  float landmarks[10] = {0};
};

struct FaceDetectorConfig {
  std::string model;
  // Optional MediaPipe face-landmark ONNX model used to refine the
  // detector's six-point output into five ArcFace-compatible points.
  std::string landmark_model;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
  int32_t input_width = 640;
  int32_t input_height = 640;
  float score_threshold = 0.5f;
  float nms_threshold = 0.4f;
  int32_t max_faces = 0;
};

class FaceDetector {
 public:
  explicit FaceDetector(const FaceDetectorConfig &config);
  ~FaceDetector();

  std::vector<FaceDetection> Detect(const ImageView &image) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Compatibility aliases for the first experimental C++ API. New code should
// use FaceDetectorConfig and FaceDetector; the model contract is generic and
// the current bundle uses MediaPipe Face Detection.
using RetinaFaceConfig = FaceDetectorConfig;
using RetinaFaceDetector = FaceDetector;

struct AuraFaceConfig {
  std::string model;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";
  int32_t input_width = 112;
  int32_t input_height = 112;
};

class AuraFaceRecognizer {
 public:
  explicit AuraFaceRecognizer(const AuraFaceConfig &config);
  ~AuraFaceRecognizer();

  int32_t Dim() const;

  // The face may be null; in that case the full image is used as a crop.
  // When landmarks are present, an ArcFace-compatible five-point alignment is
  // applied before running AuraFace.
  std::vector<float> Compute(const ImageView &image,
                             const FaceDetection *face) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

float CosineSimilarity(const float *a, const float *b, int32_t dim);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FACE_H_
