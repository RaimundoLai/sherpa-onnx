//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/face.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {
namespace {

constexpr float kArcFaceLandmarks[10] = {
    38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f,
    71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f};

struct BgrPixel {
  float b;
  float g;
  float r;
};

struct SimilarityTransform {
  float a = 1;
  float b = 0;
  float tx = 0;
  float ty = 0;
};

struct Candidate {
  FaceDetection face;
};

bool IsFinite(float v) { return std::isfinite(v); }

int32_t BytesPerPixel(const ImageView &image) {
  return image.channels;
}

BgrPixel ReadBgr(const ImageView &image, int32_t x, int32_t y) {
  x = std::max<int32_t>(0, std::min<int32_t>(image.width - 1, x));
  y = std::max<int32_t>(0, std::min<int32_t>(image.height - 1, y));
  const uint8_t *p = image.data + y * image.EffectiveStride() +
                     x * BytesPerPixel(image);

  switch (image.format) {
    case ImageFormat::kBGR:
      return {static_cast<float>(p[0]), static_cast<float>(p[1]),
              static_cast<float>(p[2])};
    case ImageFormat::kBGRA:
      return {static_cast<float>(p[0]), static_cast<float>(p[1]),
              static_cast<float>(p[2])};
    case ImageFormat::kRGBA:
      return {static_cast<float>(p[2]), static_cast<float>(p[1]),
              static_cast<float>(p[0])};
    case ImageFormat::kRGB:
    default:
      return {static_cast<float>(p[2]), static_cast<float>(p[1]),
              static_cast<float>(p[0])};
  }
}

BgrPixel SampleBgr(const ImageView &image, float x, float y) {
  x = std::max(0.0f, std::min(x, static_cast<float>(image.width - 1)));
  y = std::max(0.0f, std::min(y, static_cast<float>(image.height - 1)));
  int32_t x0 = static_cast<int32_t>(std::floor(x));
  int32_t y0 = static_cast<int32_t>(std::floor(y));
  int32_t x1 = std::min<int32_t>(x0 + 1, image.width - 1);
  int32_t y1 = std::min<int32_t>(y0 + 1, image.height - 1);
  float dx = x - x0;
  float dy = y - y0;

  BgrPixel p00 = ReadBgr(image, x0, y0);
  BgrPixel p10 = ReadBgr(image, x1, y0);
  BgrPixel p01 = ReadBgr(image, x0, y1);
  BgrPixel p11 = ReadBgr(image, x1, y1);
  auto interpolate = [dx, dy](float v00, float v10, float v01,
                              float v11) {
    float top = v00 + dx * (v10 - v00);
    float bottom = v01 + dx * (v11 - v01);
    return top + dy * (bottom - top);
  };
  return {interpolate(p00.b, p10.b, p01.b, p11.b),
          interpolate(p00.g, p10.g, p01.g, p11.g),
          interpolate(p00.r, p10.r, p01.r, p11.r)};
}

std::vector<float> ResizeToBgr(const ImageView &image, int32_t width,
                               int32_t height) {
  std::vector<float> output(static_cast<size_t>(width) * height * 3);
  for (int32_t y = 0; y != height; ++y) {
    float source_y = (y + 0.5f) * image.height / height - 0.5f;
    for (int32_t x = 0; x != width; ++x) {
      float source_x = (x + 0.5f) * image.width / width - 0.5f;
      BgrPixel p = SampleBgr(image, source_x, source_y);
      size_t offset = (static_cast<size_t>(y) * width + x) * 3;
      output[offset] = p.b;
      output[offset + 1] = p.g;
      output[offset + 2] = p.r;
    }
  }
  return output;
}

std::vector<int64_t> ShapeOf(const Ort::Value &value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

size_t ElementCount(const Ort::Value &value) {
  return value.GetTensorTypeAndShapeInfo().GetElementCount();
}

float IntersectionOverUnion(const FaceDetection &a, const FaceDetection &b) {
  float x1 = std::max(a.bbox[0], b.bbox[0]);
  float y1 = std::max(a.bbox[1], b.bbox[1]);
  float x2 = std::min(a.bbox[2], b.bbox[2]);
  float y2 = std::min(a.bbox[3], b.bbox[3]);
  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float area_a = std::max(0.0f, a.bbox[2] - a.bbox[0]) *
                std::max(0.0f, a.bbox[3] - a.bbox[1]);
  float area_b = std::max(0.0f, b.bbox[2] - b.bbox[0]) *
                std::max(0.0f, b.bbox[3] - b.bbox[1]);
  float denominator = area_a + area_b - intersection;
  return denominator > 0 ? intersection / denominator : 0;
}

std::vector<FaceDetection> ApplyNms(std::vector<FaceDetection> candidates,
                                    float threshold, int32_t max_faces) {
  std::sort(candidates.begin(), candidates.end(),
            [](const FaceDetection &a, const FaceDetection &b) {
              return a.score > b.score;
            });
  std::vector<FaceDetection> selected;
  selected.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    bool suppressed = false;
    for (const auto &face : selected) {
      if (IntersectionOverUnion(candidate, face) > threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      selected.push_back(candidate);
      if (max_faces > 0 &&
          static_cast<int32_t>(selected.size()) >= max_faces) {
        break;
      }
    }
  }
  return selected;
}

std::vector<std::array<float, 4>> GeneratePriors(int32_t width,
                                                  int32_t height) {
  constexpr int32_t kSteps[3] = {8, 16, 32};
  constexpr int32_t kMinSizes[3][2] = {{16, 32}, {64, 128}, {256, 512}};
  std::vector<std::array<float, 4>> priors;
  for (int32_t level = 0; level != 3; ++level) {
    int32_t feature_height = (height + kSteps[level] - 1) / kSteps[level];
    int32_t feature_width = (width + kSteps[level] - 1) / kSteps[level];
    for (int32_t y = 0; y != feature_height; ++y) {
      for (int32_t x = 0; x != feature_width; ++x) {
        for (int32_t k = 0; k != 2; ++k) {
          float center_x = (x + 0.5f) * kSteps[level] / width;
          float center_y = (y + 0.5f) * kSteps[level] / height;
          float box_w = static_cast<float>(kMinSizes[level][k]) / width;
          float box_h = static_cast<float>(kMinSizes[level][k]) / height;
          priors.push_back({center_x, center_y, box_w, box_h});
        }
      }
    }
  }
  return priors;
}

SimilarityTransform EstimateSimilarityTransform(const float *source) {
  double source_mean_x = 0;
  double source_mean_y = 0;
  double target_mean_x = 0;
  double target_mean_y = 0;
  for (int32_t i = 0; i != 5; ++i) {
    source_mean_x += source[2 * i];
    source_mean_y += source[2 * i + 1];
    target_mean_x += kArcFaceLandmarks[2 * i];
    target_mean_y += kArcFaceLandmarks[2 * i + 1];
  }
  source_mean_x /= 5;
  source_mean_y /= 5;
  target_mean_x /= 5;
  target_mean_y /= 5;

  double variance = 0;
  double numerator_a = 0;
  double numerator_b = 0;
  for (int32_t i = 0; i != 5; ++i) {
    double sx = source[2 * i] - source_mean_x;
    double sy = source[2 * i + 1] - source_mean_y;
    double tx = kArcFaceLandmarks[2 * i] - target_mean_x;
    double ty = kArcFaceLandmarks[2 * i + 1] - target_mean_y;
    variance += sx * sx + sy * sy;
    numerator_a += sx * tx + sy * ty;
    numerator_b += sx * ty - sy * tx;
  }
  if (variance < 1e-8) {
    throw std::invalid_argument("Face landmarks are degenerate");
  }
  SimilarityTransform transform;
  transform.a = static_cast<float>(numerator_a / variance);
  transform.b = static_cast<float>(numerator_b / variance);
  transform.tx = static_cast<float>(target_mean_x - transform.a * source_mean_x +
                                    transform.b * source_mean_y);
  transform.ty = static_cast<float>(target_mean_y - transform.b * source_mean_x -
                                    transform.a * source_mean_y);
  return transform;
}

bool HasUsableLandmarks(const FaceDetection *face) {
  if (!face) return false;
  for (float value : face->landmarks) {
    if (!IsFinite(value)) return false;
  }
  return true;
}

std::vector<float> CropAndAlign(const ImageView &image, int32_t width,
                                int32_t height, const FaceDetection *face) {
  SimilarityTransform transform;
  bool use_alignment = HasUsableLandmarks(face);
  if (use_alignment) {
    transform = EstimateSimilarityTransform(face->landmarks);
  } else {
    float x1 = 0;
    float y1 = 0;
    float x2 = static_cast<float>(image.width);
    float y2 = static_cast<float>(image.height);
    if (face) {
      x1 = face->bbox[0];
      y1 = face->bbox[1];
      x2 = face->bbox[2];
      y2 = face->bbox[3];
    }
    float side = std::max(x2 - x1, y2 - y1);
    float center_x = 0.5f * (x1 + x2);
    float center_y = 0.5f * (y1 + y2);
    float margin = 0.25f * side;
    x1 = center_x - 0.5f * side - margin;
    y1 = center_y - 0.5f * side - margin;
    side *= 1.5f;
    // q = a * p + t maps source pixels to the target square.
    transform.a = width / side;
    transform.b = 0;
    transform.tx = width * 0.5f - transform.a * center_x;
    transform.ty = height * 0.5f - transform.a * center_y;
  }

  float determinant = transform.a * transform.a + transform.b * transform.b;
  if (determinant < 1e-10f) {
    throw std::invalid_argument("Face alignment transform is degenerate");
  }

  std::vector<float> output(static_cast<size_t>(width) * height * 3);
  for (int32_t y = 0; y != height; ++y) {
    for (int32_t x = 0; x != width; ++x) {
      // Invert q = [a -b; b a]p + t.
      float u = x - transform.tx;
      float v = y - transform.ty;
      float source_x = (transform.a * u + transform.b * v) / determinant;
      float source_y = (-transform.b * u + transform.a * v) / determinant;
      BgrPixel p = SampleBgr(image, source_x, source_y);
      size_t offset = (static_cast<size_t>(y) * width + x) * 3;
      output[offset] = p.b;
      output[offset + 1] = p.g;
      output[offset + 2] = p.r;
    }
  }
  return output;
}

void FillRetinaNchwInput(const std::vector<float> &bgr, int32_t width,
                         int32_t height, std::vector<float> *input) {
  input->resize(static_cast<size_t>(width) * height * 3);
  size_t plane = static_cast<size_t>(width) * height;
  for (int32_t y = 0; y != height; ++y) {
    for (int32_t x = 0; x != width; ++x) {
      size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
      size_t index = static_cast<size_t>(y) * width + x;
      // The original PyTorch RetinaFace export uses BGR input with the
      // channel means [104, 117, 123].
      (*input)[index] = bgr[pixel] - 104.0f;
      (*input)[plane + index] = bgr[pixel + 1] - 117.0f;
      (*input)[2 * plane + index] = bgr[pixel + 2] - 123.0f;
    }
  }
}

void FillScrfdNchwInput(const std::vector<float> &bgr, int32_t width,
                        int32_t height, std::vector<float> *input) {
  input->resize(static_cast<size_t>(width) * height * 3);
  size_t plane = static_cast<size_t>(width) * height;
  for (int32_t y = 0; y != height; ++y) {
    for (int32_t x = 0; x != width; ++x) {
      size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
      size_t index = static_cast<size_t>(y) * width + x;
      // AuraFace SCRFD converts BGR to RGB and
      // applies (pixel - 127.5) / 128 before its NCHW ONNX graph.
      (*input)[index] = (bgr[pixel + 2] - 127.5f) / 128.0f;
      (*input)[plane + index] = (bgr[pixel + 1] - 127.5f) / 128.0f;
      (*input)[2 * plane + index] = (bgr[pixel] - 127.5f) / 128.0f;
    }
  }
}

std::vector<float> PrepareScrfdBgr(const ImageView &image, int32_t width,
                                   int32_t height, float *det_scale) {
  // Match AuraFace SCRFD preprocessing: preserve the
  // source aspect ratio and place the resized image at the top-left of a
  // zero-padded square canvas.
  float image_ratio = static_cast<float>(image.height) / image.width;
  float model_ratio = static_cast<float>(height) / width;
  int32_t new_width;
  int32_t new_height;
  if (image_ratio > model_ratio) {
    new_height = height;
    new_width = std::max(1, static_cast<int32_t>(new_height / image_ratio));
    *det_scale = static_cast<float>(new_height) / image.height;
  } else {
    new_width = width;
    new_height = std::max(1, static_cast<int32_t>(new_width * image_ratio));
    *det_scale = static_cast<float>(new_width) / image.width;
  }
  std::vector<float> resized = ResizeToBgr(image, new_width, new_height);
  std::vector<float> output(static_cast<size_t>(width) * height * 3, 0.0f);
  for (int32_t y = 0; y != new_height; ++y) {
    std::memcpy(output.data() + static_cast<size_t>(y) * width * 3,
                resized.data() + static_cast<size_t>(y) * new_width * 3,
                static_cast<size_t>(new_width) * 3 * sizeof(float));
  }
  return output;
}

struct MediaPipePreprocessedImage {
  std::vector<float> bgr;
  float scale = 1.0f;
  // Offset of the detector canvas inside the resized source. MediaPipe's
  // BlazeFace reference path uses a centered crop, not letterboxing. A
  // positive value means that this many resized source pixels were removed
  // from the left/top before inference.
  int32_t crop_x = 0;
  int32_t crop_y = 0;
};

MediaPipePreprocessedImage PrepareMediaPipeBgr(const ImageView &image,
                                               int32_t width,
                                               int32_t height) {
  // Match the MediaPipe/BlazeFace reference implementation used by
  // ComfyUI-LivePortraitKJ: resize the long side and take a centered square
  // crop. Letterboxing changes the anchor geometry and produces an overly
  // wide/tall face box on portrait sources.
  const float scale = std::max(static_cast<float>(width) / image.width,
                               static_cast<float>(height) / image.height);
  const int32_t new_width = std::max(
      1, static_cast<int32_t>(std::round(image.width * scale)));
  const int32_t new_height = std::max(
      1, static_cast<int32_t>(std::round(image.height * scale)));
  const int32_t crop_x = std::max(0, (new_width - width) / 2);
  const int32_t crop_y = std::max(0, (new_height - height) / 2);
  std::vector<float> resized = ResizeToBgr(image, new_width, new_height);
  std::vector<float> output(static_cast<size_t>(width) * height * 3);
  for (int32_t y = 0; y != new_height; ++y) {
    if (y < crop_y || y >= crop_y + height) continue;
    std::memcpy(output.data() + static_cast<size_t>(y - crop_y) * width * 3,
                resized.data() +
                    (static_cast<size_t>(y) * new_width + crop_x) * 3,
                static_cast<size_t>(width) * 3 * sizeof(float));
  }
  return {std::move(output), scale, crop_x, crop_y};
}

void FillMediaPipeNchwInput(const std::vector<float> &bgr, int32_t width,
                            int32_t height, std::vector<float> *input) {
  input->resize(static_cast<size_t>(width) * height * 3);
  const size_t plane = static_cast<size_t>(width) * height;
  for (int32_t y = 0; y != height; ++y) {
    for (int32_t x = 0; x != width; ++x) {
      const size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
      const size_t index = static_cast<size_t>(y) * width + x;
      // BlazeFace uses RGB values in [-1, 1], matching the reference
      // implementation in ComfyUI-LivePortraitKJ.
      (*input)[index] = bgr[pixel + 2] / 127.5f - 1.0f;
      (*input)[plane + index] = bgr[pixel + 1] / 127.5f - 1.0f;
      (*input)[2 * plane + index] = bgr[pixel] / 127.5f - 1.0f;
    }
  }
}

std::vector<std::array<float, 4>> GenerateMediaPipeAnchors() {
  // face_detection_front.pbtxt: four layers with strides 8, 16, 16, 16,
  // one aspect ratio, an interpolated anchor, and fixed anchor size.
  constexpr int32_t kInputSize = 128;
  constexpr int32_t kStrides[4] = {8, 16, 16, 16};
  constexpr float kMinScale = 0.1484375f;
  constexpr float kMaxScale = 0.75f;
  std::vector<std::array<float, 4>> anchors;
  int32_t layer = 0;
  while (layer < 4) {
    int32_t last = layer;
    std::vector<float> scales;
    while (last < 4 && kStrides[last] == kStrides[layer]) {
      const float scale = kMinScale +
                          (kMaxScale - kMinScale) * last / 3.0f;
      scales.push_back(scale);
      const float next_scale =
          last == 3 ? 1.0f
                    : kMinScale + (kMaxScale - kMinScale) * (last + 1) / 3.0f;
      scales.push_back(std::sqrt(scale * next_scale));
      ++last;
    }
    const int32_t feature =
        (kInputSize + kStrides[layer] - 1) / kStrides[layer];
    for (int32_t y = 0; y != feature; ++y) {
      for (int32_t x = 0; x != feature; ++x) {
        for (float scale : scales) {
          (void)scale;  // The exported detector uses fixed anchor size.
          anchors.push_back({(x + 0.5f) / feature,
                             (y + 0.5f) / feature, 1.0f, 1.0f});
        }
      }
    }
    layer = last;
  }
  return anchors;
}

void FillSerengilNhwcInput(const std::vector<float> &bgr,
                           std::vector<float> *input) {
  // serengil/retinaface preprocesses a BGR image into raw RGB NHWC pixels.
  input->resize(bgr.size());
  for (size_t i = 0; i != bgr.size(); i += 3) {
    (*input)[i] = bgr[i + 2];
    (*input)[i + 1] = bgr[i + 1];
    (*input)[i + 2] = bgr[i];
  }
}

void FillAuraNchwInput(const std::vector<float> &bgr, int32_t width,
                       int32_t height, std::vector<float> *input) {
  input->resize(static_cast<size_t>(width) * height * 3);
  size_t plane = static_cast<size_t>(width) * height;
  for (int32_t y = 0; y != height; ++y) {
    for (int32_t x = 0; x != width; ++x) {
      size_t pixel = (static_cast<size_t>(y) * width + x) * 3;
      size_t index = static_cast<size_t>(y) * width + x;
      // InsightFace's ArcFace ONNX path converts BGR to RGB and applies
      // (pixel - 127.5) / 127.5 before inference.
      (*input)[index] = (bgr[pixel + 2] - 127.5f) / 127.5f;
      (*input)[plane + index] = (bgr[pixel + 1] - 127.5f) / 127.5f;
      (*input)[2 * plane + index] = (bgr[pixel] - 127.5f) / 127.5f;
    }
  }
}

}  // namespace

bool ImageView::IsValid() const {
  if (!data || width <= 0 || height <= 0) return false;
  if (channels != 3 && channels != 4) return false;
  int32_t format_value = static_cast<int32_t>(format);
  if (format_value < 0 || format_value > 3) return false;
  if (format == ImageFormat::kRGB && channels != 3) return false;
  if (format == ImageFormat::kBGR && channels != 3) return false;
  if (format == ImageFormat::kRGBA && channels != 4) return false;
  if (format == ImageFormat::kBGRA && channels != 4) return false;
  return EffectiveStride() >= width * channels;
}

int32_t ImageView::EffectiveStride() const {
  return stride > 0 ? stride : width * channels;
}

// CoreML's provider factory can touch ONNX Runtime's default logger while
// SessionOptions are being built. Ensure the shared environment exists first.
Ort::SessionOptions GetFaceSessionOptions(int32_t num_threads,
                                           const std::string &provider) {
  auto &ort_env = GetOrtEnv();
  (void)ort_env;
  return GetSessionOptions(num_threads, provider);
}

class FaceDetector::Impl {
 public:
  explicit Impl(const FaceDetectorConfig &config)
      : config_(config),
        session_options_(GetFaceSessionOptions(config.num_threads,
                                               config.provider)),
        allocator_{} {
    if (config_.model.empty() || !FileExists(config_.model)) {
      throw std::invalid_argument("Face detector model does not exist");
    }
    session_ = std::make_unique<Ort::Session>(
        GetOrtEnv(), SHERPA_ONNX_TO_ORT_PATH(config_.model),
        session_options_);
    GetInputNames(session_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(session_.get(), &output_names_, &output_names_ptr_);

    auto shape = session_->GetInputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    if (shape.size() != 4) {
      throw std::invalid_argument("Face detector expects a rank-4 input");
    }
    nhwc_ = shape.back() == 3;
    if (!nhwc_ && shape[1] != 3) {
      throw std::invalid_argument("Face detector input must be NCHW or NHWC");
    }
    if (nhwc_) {
      if (shape[2] > 0) config_.input_width = shape[2];
      if (shape[1] > 0) config_.input_height = shape[1];
    } else {
      if (shape[3] > 0) config_.input_width = shape[3];
      if (shape[2] > 0) config_.input_height = shape[2];
    }
    if (config_.input_width <= 0 || config_.input_height <= 0) {
      throw std::invalid_argument("Face detector input dimensions are invalid");
    }
    serengil_ = output_names_.size() == 9 && nhwc_;
    scrfd_onnx_ = output_names_.size() == 9 && !nhwc_;
    mediapipe_onnx_ = output_names_.size() == 4 &&
                      HasOutput("box_coords_1") &&
                      HasOutput("box_coords_2") &&
                      HasOutput("box_scores_1") &&
                      HasOutput("box_scores_2");
    dynamic_input_ = nhwc_ && (shape[1] <= 0 || shape[2] <= 0);

    if (!config_.landmark_model.empty()) {
      if (!FileExists(config_.landmark_model)) {
        throw std::invalid_argument(
            "MediaPipe face landmark model does not exist");
      }
      auto landmark_session_options =
          GetFaceSessionOptions(config.num_threads, config.provider);
      landmark_session_ = std::make_unique<Ort::Session>(
          GetOrtEnv(), SHERPA_ONNX_TO_ORT_PATH(config_.landmark_model),
          landmark_session_options);
      GetInputNames(landmark_session_.get(), &landmark_input_names_,
                    &landmark_input_names_ptr_);
      GetOutputNames(landmark_session_.get(), &landmark_output_names_,
                     &landmark_output_names_ptr_);
      auto landmark_shape = landmark_session_->GetInputTypeInfo(0)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
      if (landmark_shape.size() != 4 || landmark_shape[1] != 3 ||
          landmark_shape[2] <= 0 || landmark_shape[3] <= 0) {
        throw std::invalid_argument(
            "MediaPipe face landmark model must use NCHW input");
      }
      landmark_input_width_ = landmark_shape[3];
      landmark_input_height_ = landmark_shape[2];
    }
  }

  std::vector<FaceDetection> Detect(const ImageView &image) const {
    if (!image.IsValid()) {
      throw std::invalid_argument("Invalid image view");
    }
    int32_t input_width = config_.input_width;
    int32_t input_height = config_.input_height;
    if (serengil_ && dynamic_input_) {
      // Match serengil/retinaface's resize_image(): target the short side at
      // 1024 pixels and cap the long side at 1980 pixels.
      int32_t short_side = std::min(image.width, image.height);
      int32_t long_side = std::max(image.width, image.height);
      float scale = 1024.0f / short_side;
      if (std::round(scale * long_side) > 1980.0f) {
        scale = 1980.0f / long_side;
      }
      input_width = std::max(1, static_cast<int32_t>(std::round(image.width * scale)));
      input_height =
          std::max(1, static_cast<int32_t>(std::round(image.height * scale)));
    }
    float det_scale = 1.0f;
    int32_t media_crop_x = 0;
    int32_t media_crop_y = 0;
    std::vector<float> bgr;
    if (mediapipe_onnx_) {
      auto prepared = PrepareMediaPipeBgr(image, input_width, input_height);
      det_scale = prepared.scale;
      media_crop_x = prepared.crop_x;
      media_crop_y = prepared.crop_y;
      bgr = std::move(prepared.bgr);
    } else {
      bgr = scrfd_onnx_ ? PrepareScrfdBgr(image, input_width, input_height,
                                          &det_scale)
                        : ResizeToBgr(image, input_width, input_height);
    }
    std::vector<float> input;
    std::vector<int64_t> shape;
    if (mediapipe_onnx_) {
      if (nhwc_) {
        throw std::invalid_argument(
            "MediaPipe face detector model must use NCHW input");
      }
      FillMediaPipeNchwInput(bgr, input_width, input_height, &input);
      shape = {1, 3, input_height, input_width};
    } else if (serengil_ && nhwc_) {
      FillSerengilNhwcInput(bgr, &input);
      shape = {1, input_height, input_width, 3};
    } else if (scrfd_onnx_) {
      FillScrfdNchwInput(bgr, input_width, input_height, &input);
      shape = {1, 3, input_height, input_width};
    } else {
      if (nhwc_) {
        throw std::invalid_argument(
            "Three-output face detector models must use NCHW input");
      }
      FillRetinaNchwInput(bgr, input_width, input_height, &input);
      shape = {1, 3, input_height, input_width};
    }
    auto tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        input.data(), input.size(), shape.data(), shape.size());
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names_ptr_.data(), &tensor, 1,
                                 output_names_ptr_.data(), output_names_ptr_.size());
    if (outputs.size() == 4 && mediapipe_onnx_) {
      auto faces = DecodeMediaPipeOutputs(outputs, image, input_width,
                                          input_height, media_crop_x,
                                          media_crop_y, det_scale);
      if (landmark_session_) RefineMediaPipeLandmarks(image, &faces);
      return faces;
    }
    if (outputs.size() == 3) {
      return DecodeStandardOutputs(outputs, image, input_width, input_height);
    }
    if (outputs.size() == 9) {
      if (scrfd_onnx_) {
        return DecodeScrfdOutputs(outputs, image, input_width,
                                               input_height, 1.0f / det_scale,
                                               1.0f / det_scale);
      }
      return DecodeSerengilOutputs(outputs, image, input_width, input_height);
    }
    throw std::invalid_argument(
        "Face detector ONNX model must expose MediaPipe (4), RetinaFace (3), "
        "or TensorFlow/SCRFD (9) outputs");
  }

 private:
  bool HasOutput(const char *name) const {
    return std::find(output_names_.begin(), output_names_.end(), name) !=
           output_names_.end();
  }

  size_t OutputIndex(const char *name) const {
    auto it = std::find(output_names_.begin(), output_names_.end(), name);
    if (it == output_names_.end()) {
      throw std::invalid_argument(std::string("Missing MediaPipe output: ") +
                                  name);
    }
    return static_cast<size_t>(it - output_names_.begin());
  }

  std::vector<FaceDetection> DecodeMediaPipeOutputs(
      const std::vector<Ort::Value> &outputs, const ImageView &image,
      int32_t input_width, int32_t input_height, int32_t pad_x,
      int32_t pad_y, float det_scale) const {
    const size_t coords1 = OutputIndex("box_coords_1");
    const size_t coords2 = OutputIndex("box_coords_2");
    const size_t scores1 = OutputIndex("box_scores_1");
    const size_t scores2 = OutputIndex("box_scores_2");
    const auto anchors = GenerateMediaPipeAnchors();
    if (anchors.size() != 896) {
      throw std::invalid_argument("MediaPipe anchor count is invalid");
    }
    std::vector<FaceDetection> candidates;
    auto decode = [&](size_t coords_index, size_t scores_index,
                      size_t anchor_offset) {
      const auto coord_shape = ShapeOf(outputs[coords_index]);
      const auto score_shape = ShapeOf(outputs[scores_index]);
      if (coord_shape.size() != 3 || score_shape.size() != 3 ||
          coord_shape[0] != 1 || score_shape[0] != 1 ||
          coord_shape[2] != 16 || score_shape[2] != 1 ||
          coord_shape[1] != score_shape[1]) {
        throw std::invalid_argument("Invalid MediaPipe detector outputs");
      }
      const size_t count = static_cast<size_t>(coord_shape[1]);
      const float *coords = outputs[coords_index].GetTensorData<float>();
      const float *scores = outputs[scores_index].GetTensorData<float>();
      // The bundled Heliosoph model is the 256x256 BlazeFace back model.
      // Its raw coordinates use the input-size scale (the 128 model uses
      // 128); anchors remain normalized and have unit width/height.
      const float coordinate_scale = input_width <= 128 ? 128.0f : 256.0f;
      for (size_t i = 0; i != count; ++i) {
        const float score = 1.0f / (1.0f + std::exp(-std::max(
            -100.0f, std::min(100.0f, scores[i]))));
        if (score < config_.score_threshold) continue;
        const auto &anchor = anchors[anchor_offset + i];
        const float *raw = coords + i * 16;
        const float center_x = raw[0] / coordinate_scale + anchor[0];
        const float center_y = raw[1] / coordinate_scale + anchor[1];
        const float box_w = raw[2] / coordinate_scale;
        const float box_h = raw[3] / coordinate_scale;
        auto map_x = [&](float normalized) {
          return (normalized * input_width + pad_x) / det_scale;
        };
        auto map_y = [&](float normalized) {
          return (normalized * input_height + pad_y) / det_scale;
        };
        FaceDetection face;
        face.score = score;
        face.bbox[0] = std::max(0.0f, map_x(center_x - box_w * 0.5f));
        face.bbox[1] = std::max(0.0f, map_y(center_y - box_h * 0.5f));
        face.bbox[2] = std::min(static_cast<float>(image.width),
                                map_x(center_x + box_w * 0.5f));
        face.bbox[3] = std::min(static_cast<float>(image.height),
                                map_y(center_y + box_h * 0.5f));
        for (int32_t keypoint = 0; keypoint != 5; ++keypoint) {
          face.landmarks[2 * keypoint] = map_x(
              raw[4 + keypoint * 2] / coordinate_scale + anchor[0]);
          face.landmarks[2 * keypoint + 1] = map_y(
              raw[5 + keypoint * 2] / coordinate_scale + anchor[1]);
        }
        candidates.push_back(face);
      }
    };
    decode(coords1, scores1, 0);
    decode(coords2, scores2, 512);
    return ApplyNms(std::move(candidates), config_.nms_threshold,
                    config_.max_faces);
  }

  std::vector<float> CropMediaPipeFace(const ImageView &image,
                                       const FaceDetection &face) const {
    const float x1 = face.bbox[0];
    const float y1 = face.bbox[1];
    const float x2 = face.bbox[2];
    const float y2 = face.bbox[3];
    const float side = std::max(1.0f, std::max(x2 - x1, y2 - y1) * 1.5f);
    const float center_x = (x1 + x2) * 0.5f;
    const float center_y = (y1 + y2) * 0.5f;
    std::vector<float> output(static_cast<size_t>(landmark_input_width_) *
                              landmark_input_height_ * 3);
    for (int32_t y = 0; y != landmark_input_height_; ++y) {
      const float source_y = center_y +
          ((y + 0.5f) / landmark_input_height_ - 0.5f) * side;
      for (int32_t x = 0; x != landmark_input_width_; ++x) {
        const float source_x = center_x +
            ((x + 0.5f) / landmark_input_width_ - 0.5f) * side;
        const BgrPixel pixel = SampleBgr(image, source_x, source_y);
        const size_t offset = (static_cast<size_t>(y) * landmark_input_width_ +
                               x) * 3;
        output[offset] = pixel.b;
        output[offset + 1] = pixel.g;
        output[offset + 2] = pixel.r;
      }
    }
    return output;
  }

  void RefineMediaPipeLandmarks(const ImageView &image,
                                std::vector<FaceDetection> *faces) const {
    const size_t landmarks_index = [&]() {
      for (size_t i = 0; i != landmark_output_names_.size(); ++i) {
        if (landmark_output_names_[i] == "landmarks") return i;
      }
      throw std::invalid_argument("MediaPipe landmark output is missing");
    }();
    for (auto &face : *faces) {
      std::vector<float> crop = CropMediaPipeFace(image, face);
      std::vector<float> input;
      FillMediaPipeNchwInput(crop, landmark_input_width_, landmark_input_height_,
                             &input);
      const std::array<int64_t, 4> shape = {
          1, 3, landmark_input_height_, landmark_input_width_};
      auto tensor = Ort::Value::CreateTensor<float>(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
          input.data(), input.size(), shape.data(), shape.size());
      auto outputs = landmark_session_->Run(
          Ort::RunOptions{nullptr}, landmark_input_names_ptr_.data(), &tensor,
          1, landmark_output_names_ptr_.data(), landmark_output_names_ptr_.size());
      const auto output_shape = ShapeOf(outputs[landmarks_index]);
      if (output_shape.size() != 3 || output_shape[0] != 1 ||
          output_shape[1] != 468 || output_shape[2] < 2) {
        throw std::invalid_argument("Invalid MediaPipe landmark output shape");
      }
      const float *landmarks =
          outputs[landmarks_index].GetTensorData<float>();
      const float side = std::max(1.0f, std::max(face.bbox[2] - face.bbox[0],
                                                  face.bbox[3] - face.bbox[1]) *
                                                    1.5f);
      const float center_x = (face.bbox[0] + face.bbox[2]) * 0.5f;
      const float center_y = (face.bbox[1] + face.bbox[3]) * 0.5f;
      constexpr int32_t kIndices[5] = {33, 263, 1, 61, 291};
      bool usable = true;
      for (int32_t point = 0; point != 5; ++point) {
        const float normalized_x = landmarks[kIndices[point] * 3];
        const float normalized_y = landmarks[kIndices[point] * 3 + 1];
        if (!IsFinite(normalized_x) || !IsFinite(normalized_y) ||
            normalized_x < -0.25f || normalized_x > 1.25f ||
            normalized_y < -0.25f || normalized_y > 1.25f) {
          usable = false;
          break;
        }
        face.landmarks[point * 2] = center_x +
            (normalized_x - 0.5f) * side;
        face.landmarks[point * 2 + 1] = center_y +
            (normalized_y - 0.5f) * side;
      }
      if (!usable) continue;
    }
  }

  std::vector<FaceDetection> DecodeStandardOutputs(
      const std::vector<Ort::Value> &outputs, const ImageView &image,
      int32_t input_width, int32_t input_height) const {
    size_t loc_count = ElementCount(outputs[0]);
    size_t n = loc_count / 4;
    if (n == 0 || loc_count != n * 4 || ElementCount(outputs[2]) != n * 10) {
      throw std::invalid_argument("Invalid face detector output shapes");
    }
    size_t conf_count = ElementCount(outputs[1]);
    size_t classes = conf_count / n;
    if (classes < 1 || conf_count != n * classes || classes > 2) {
      throw std::invalid_argument("Invalid face detector confidence output");
    }
    auto priors = GeneratePriors(input_width, input_height);
    if (priors.size() != n) {
      throw std::invalid_argument(
          "Face detector output count does not match the standard priors");
    }
    const float *loc = outputs[0].GetTensorData<float>();
    const float *conf = outputs[1].GetTensorData<float>();
    const float *landmarks = outputs[2].GetTensorData<float>();
    float scale_x = static_cast<float>(image.width) / input_width;
    float scale_y = static_cast<float>(image.height) / input_height;
    std::vector<FaceDetection> candidates;
    for (size_t i = 0; i != n; ++i) {
      float score = classes == 1 ? conf[i] : conf[i * classes + 1];
      if (score < config_.score_threshold) continue;
      const auto &prior = priors[i];
      float center_x = prior[0] + loc[i * 4] * 0.1f * prior[2];
      float center_y = prior[1] + loc[i * 4 + 1] * 0.1f * prior[3];
      float box_w = prior[2] * std::exp(loc[i * 4 + 2] * 0.2f);
      float box_h = prior[3] * std::exp(loc[i * 4 + 3] * 0.2f);
      FaceDetection face;
      face.score = score;
      face.bbox[0] = std::max(0.0f, (center_x - box_w * 0.5f) *
                                      input_width * scale_x);
      face.bbox[1] = std::max(0.0f, (center_y - box_h * 0.5f) *
                                      input_height * scale_y);
      face.bbox[2] = std::min(static_cast<float>(image.width),
                              (center_x + box_w * 0.5f) *
                                  input_width * scale_x);
      face.bbox[3] = std::min(static_cast<float>(image.height),
                              (center_y + box_h * 0.5f) *
                                  input_height * scale_y);
      for (int32_t j = 0; j != 5; ++j) {
        face.landmarks[2 * j] =
            (prior[0] + landmarks[i * 10 + 2 * j] * 0.1f * prior[2]) *
            input_width * scale_x;
        face.landmarks[2 * j + 1] =
            (prior[1] + landmarks[i * 10 + 2 * j + 1] * 0.1f * prior[3]) *
            input_height * scale_y;
      }
      candidates.push_back(face);
    }
    return ApplyNms(std::move(candidates), config_.nms_threshold,
                    config_.max_faces);
  }

  std::vector<FaceDetection> DecodeSerengilOutputs(
      const std::vector<Ort::Value> &outputs, const ImageView &image,
      int32_t input_width, int32_t input_height) const {
    // This is the output order used by serengil/retinaface's TensorFlow graph:
    // score, box, landmarks for strides 32, 16 and 8.  The tensors are NHWC.
    constexpr int32_t kSteps[3] = {32, 16, 8};
    constexpr float kAnchors[3][2][4] = {
        {{-248, -248, 263, 263}, {-120, -120, 135, 135}},
        {{-56, -56, 71, 71}, {-24, -24, 39, 39}},
        {{-8, -8, 23, 23}, {0, 0, 15, 15}}};
    std::vector<FaceDetection> candidates;
    float scale_x = static_cast<float>(image.width) / input_width;
    float scale_y = static_cast<float>(image.height) / input_height;
    for (int32_t level = 0; level != 3; ++level) {
      auto score_shape = ShapeOf(outputs[level * 3]);
      auto box_shape = ShapeOf(outputs[level * 3 + 1]);
      auto landmark_shape = ShapeOf(outputs[level * 3 + 2]);
      if (score_shape.size() != 4 || box_shape.size() != 4 ||
          landmark_shape.size() != 4 || score_shape[0] != 1 ||
          box_shape[0] != 1 || landmark_shape[0] != 1) {
        throw std::invalid_argument("Invalid RetinaFace TensorFlow outputs");
      }
      int32_t h = static_cast<int32_t>(score_shape[1]);
      int32_t w = static_cast<int32_t>(score_shape[2]);
      int32_t anchors = 2;
      if (score_shape[3] != 2 * anchors ||
          box_shape[1] != h || box_shape[2] != w ||
          box_shape[3] != 4 * anchors || landmark_shape[1] != h ||
          landmark_shape[2] != w || landmark_shape[3] != 10 * anchors) {
        throw std::invalid_argument("Unsupported RetinaFace TensorFlow layout");
      }
      const float *scores = outputs[level * 3].GetTensorData<float>();
      const float *boxes = outputs[level * 3 + 1].GetTensorData<float>();
      const float *landmarks = outputs[level * 3 + 2].GetTensorData<float>();
      float stride = static_cast<float>(kSteps[level]);
      for (int32_t y = 0; y != h; ++y) {
        for (int32_t x = 0; x != w; ++x) {
          for (int32_t a = 0; a != anchors; ++a) {
            size_t pixel = static_cast<size_t>(y) * w + x;
            float score = scores[pixel * 2 * anchors + anchors + a];
            if (score < config_.score_threshold) continue;
            const float *anchor = kAnchors[level][a];
            float anchor_x1 = x * stride + anchor[0];
            float anchor_y1 = y * stride + anchor[1];
            float anchor_x2 = x * stride + anchor[2];
            float anchor_y2 = y * stride + anchor[3];
            float anchor_cx = (anchor_x1 + anchor_x2) * 0.5f;
            float anchor_cy = (anchor_y1 + anchor_y2) * 0.5f;
            float anchor_w = anchor_x2 - anchor_x1 + 1;
            float anchor_h = anchor_y2 - anchor_y1 + 1;
            const float *box = boxes + pixel * 4 * anchors + a * 4;
            FaceDetection face;
            face.score = score;
            float cx = box[0] * anchor_w + anchor_cx;
            float cy = box[1] * anchor_h + anchor_cy;
            float bw = std::exp(box[2]) * anchor_w;
            float bh = std::exp(box[3]) * anchor_h;
            // Match postprocess.bbox_pred(): anchor coordinates are treated
            // as inclusive, hence the -1 in the decoded width and height.
            face.bbox[0] =
                std::max(0.0f, (cx - 0.5f * (bw - 1.0f)) * scale_x);
            face.bbox[1] =
                std::max(0.0f, (cy - 0.5f * (bh - 1.0f)) * scale_y);
            face.bbox[2] = std::min(static_cast<float>(image.width),
                                    (cx + 0.5f * (bw - 1.0f)) * scale_x);
            face.bbox[3] = std::min(static_cast<float>(image.height),
                                    (cy + 0.5f * (bh - 1.0f)) * scale_y);
            const float *landmark =
                landmarks + pixel * 10 * anchors + a * 10;
            for (int32_t j = 0; j != 5; ++j) {
              face.landmarks[2 * j] =
                  (landmark[2 * j] * anchor_w + anchor_cx) * scale_x;
              face.landmarks[2 * j + 1] =
                  (landmark[2 * j + 1] * anchor_h + anchor_cy) * scale_y;
            }
            candidates.push_back(face);
          }
        }
      }
    }
    return ApplyNms(std::move(candidates), config_.nms_threshold,
                    config_.max_faces);
  }

  std::vector<FaceDetection> DecodeScrfdOutputs(
      const std::vector<Ort::Value> &outputs, const ImageView &image,
      int32_t input_width, int32_t input_height, float scale_x,
      float scale_y) const {
    // AuraFace SCRFD exposes nine flattened NCHW outputs: scores for the
    // three FPN levels, then box distances, then five-point landmark
    // distances. Each FPN location has two anchors.
    constexpr int32_t kSteps[3] = {8, 16, 32};
    std::vector<FaceDetection> candidates;
    for (int32_t level = 0; level != 3; ++level) {
      auto score_shape = ShapeOf(outputs[level]);
      auto box_shape = ShapeOf(outputs[level + 3]);
      auto landmark_shape = ShapeOf(outputs[level + 6]);
      if (score_shape.size() != 2 || box_shape.size() != 2 ||
          landmark_shape.size() != 2 || score_shape[1] != 1 ||
          box_shape[1] != 4 || landmark_shape[1] != 10 ||
          score_shape[0] != box_shape[0] ||
          score_shape[0] != landmark_shape[0]) {
        throw std::invalid_argument(
            "Unsupported SCRFD output layout");
      }
      int32_t h = (input_height + kSteps[level] - 1) / kSteps[level];
      int32_t w = (input_width + kSteps[level] - 1) / kSteps[level];
      size_t expected = static_cast<size_t>(h) * w * 2;
      if (static_cast<size_t>(score_shape[0]) != expected) {
        throw std::invalid_argument(
            "SCRFD output count does not match its "
            "feature map");
      }
      const float *scores = outputs[level].GetTensorData<float>();
      const float *boxes = outputs[level + 3].GetTensorData<float>();
      const float *landmarks = outputs[level + 6].GetTensorData<float>();
      float stride = static_cast<float>(kSteps[level]);
      for (int32_t y = 0; y != h; ++y) {
        for (int32_t x = 0; x != w; ++x) {
          for (int32_t a = 0; a != 2; ++a) {
            size_t i =
                (static_cast<size_t>(y) * w + x) * 2 + a;
            float score = scores[i];
            if (score < config_.score_threshold) continue;
            float center_x = x * stride;
            float center_y = y * stride;
            const float *box = boxes + i * 4;
            FaceDetection face;
            face.score = score;
            face.bbox[0] = std::max(0.0f,
                                    (center_x - box[0] * stride) * scale_x);
            face.bbox[1] = std::max(0.0f,
                                    (center_y - box[1] * stride) * scale_y);
            face.bbox[2] = std::min(static_cast<float>(image.width),
                                    (center_x + box[2] * stride) * scale_x);
            face.bbox[3] = std::min(static_cast<float>(image.height),
                                    (center_y + box[3] * stride) * scale_y);
            const float *landmark = landmarks + i * 10;
            for (int32_t j = 0; j != 5; ++j) {
              face.landmarks[2 * j] =
                  (center_x + landmark[2 * j] * stride) * scale_x;
              face.landmarks[2 * j + 1] =
                  (center_y + landmark[2 * j + 1] * stride) * scale_y;
            }
            candidates.push_back(face);
          }
        }
      }
    }
    return ApplyNms(std::move(candidates), config_.nms_threshold,
                    config_.max_faces);
  }

 private:
  FaceDetectorConfig config_;
  bool nhwc_ = false;
  bool serengil_ = false;
  bool scrfd_onnx_ = false;
  bool mediapipe_onnx_ = false;
  bool dynamic_input_ = false;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::Session> landmark_session_;
  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
  int32_t landmark_input_width_ = 192;
  int32_t landmark_input_height_ = 192;
  std::vector<std::string> landmark_input_names_;
  std::vector<const char *> landmark_input_names_ptr_;
  std::vector<std::string> landmark_output_names_;
  std::vector<const char *> landmark_output_names_ptr_;
};

FaceDetector::FaceDetector(const FaceDetectorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FaceDetector::~FaceDetector() = default;

std::vector<FaceDetection> FaceDetector::Detect(
    const ImageView &image) const {
  return impl_->Detect(image);
}

class AuraFaceRecognizer::Impl {
 public:
  explicit Impl(const AuraFaceConfig &config)
      : config_(config),
        session_options_(GetFaceSessionOptions(config.num_threads,
                                               config.provider)),
        allocator_{} {
    if (config_.model.empty() || !FileExists(config_.model)) {
      throw std::invalid_argument("AuraFace model does not exist");
    }
    session_ = std::make_unique<Ort::Session>(
        GetOrtEnv(), SHERPA_ONNX_TO_ORT_PATH(config_.model),
        session_options_);
    GetInputNames(session_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(session_.get(), &output_names_, &output_names_ptr_);
    auto shape = session_->GetInputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    if (shape.size() != 4 || (shape[1] > 0 && shape[1] != 3)) {
      throw std::invalid_argument("AuraFace expects a rank-4 3-channel input");
    }
    if (shape[2] > 0) config_.input_height = shape[2];
    if (shape[3] > 0) config_.input_width = shape[3];
    if (config_.input_width <= 0 || config_.input_height <= 0) {
      throw std::invalid_argument("AuraFace input dimensions are invalid");
    }
    dim_ = static_cast<int32_t>(
        session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount());
    if (dim_ <= 0) throw std::invalid_argument("AuraFace output is empty");
  }

  int32_t Dim() const { return dim_; }

  std::vector<float> Compute(const ImageView &image,
                             const FaceDetection *face) const {
    if (!image.IsValid()) throw std::invalid_argument("Invalid image view");
    std::vector<float> bgr =
        CropAndAlign(image, config_.input_width, config_.input_height, face);
    std::vector<float> input;
    FillAuraNchwInput(bgr, config_.input_width, config_.input_height, &input);
    std::vector<int64_t> shape = {1, 3, config_.input_height,
                                  config_.input_width};
    auto tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        input.data(), input.size(), shape.data(), shape.size());
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names_ptr_.data(), &tensor, 1,
                                 output_names_ptr_.data(), output_names_ptr_.size());
    if (outputs.empty() || ElementCount(outputs[0]) !=
                               static_cast<size_t>(dim_)) {
      throw std::invalid_argument("Invalid AuraFace output");
    }
    const float *raw = outputs[0].GetTensorData<float>();
    std::vector<float> embedding(raw, raw + dim_);
    float norm = std::sqrt(std::inner_product(embedding.begin(), embedding.end(),
                                              embedding.begin(), 0.0f));
    if (norm > 1e-12f) {
      for (float &v : embedding) v /= norm;
    }
    return embedding;
  }

 private:
  AuraFaceConfig config_;
  int32_t dim_ = 0;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

AuraFaceRecognizer::AuraFaceRecognizer(const AuraFaceConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

AuraFaceRecognizer::~AuraFaceRecognizer() = default;

int32_t AuraFaceRecognizer::Dim() const { return impl_->Dim(); }

std::vector<float> AuraFaceRecognizer::Compute(
    const ImageView &image, const FaceDetection *face) const {
  return impl_->Compute(image, face);
}

float CosineSimilarity(const float *a, const float *b, int32_t dim) {
  if (!a || !b || dim <= 0) return 0;
  double dot = 0;
  double norm_a = 0;
  double norm_b = 0;
  for (int32_t i = 0; i != dim; ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    norm_a += static_cast<double>(a[i]) * a[i];
    norm_b += static_cast<double>(b[i]) * b[i];
  }
  if (norm_a <= 1e-20 || norm_b <= 1e-20) return 0;
  return static_cast<float>(dot / std::sqrt(norm_a * norm_b));
}

}  // namespace sherpa_onnx
