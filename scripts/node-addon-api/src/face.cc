//
// Copyright (c)  2026  Xiaomi Corporation

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

namespace {

struct ParsedImage {
  Napi::Value data_holder;
  SherpaOnnxImage image{};
};

void ThrowTypeError(const Napi::Env &env, const std::string &message) {
  Napi::TypeError::New(env, message).ThrowAsJavaScriptException();
}

bool ReadInt(const Napi::Object &object, const char *name, int32_t *value) {
  if (!object.Has(name) || !object.Get(name).IsNumber()) return false;
  *value = object.Get(name).As<Napi::Number>().Int32Value();
  return true;
}

bool ReadFloat(const Napi::Object &object, const char *name, float *value) {
  if (!object.Has(name) || !object.Get(name).IsNumber()) return false;
  *value = object.Get(name).As<Napi::Number>().FloatValue();
  return true;
}

int32_t ParseImageFormat(const Napi::Env &env, const Napi::Value &value) {
  if (value.IsUndefined() || value.IsNull()) return 0;
  if (value.IsNumber()) {
    int32_t format = value.As<Napi::Number>().Int32Value();
    if (format >= 0 && format <= 3) return format;
  } else if (value.IsString()) {
    std::string format = value.As<Napi::String>().Utf8Value();
    if (format == "rgb") return 0;
    if (format == "bgr") return 1;
    if (format == "rgba") return 2;
    if (format == "bgra") return 3;
  }
  ThrowTypeError(env, "image.format must be rgb, bgr, rgba, bgra, or 0..3");
  return -1;
}

bool ParseImage(const Napi::Env &env, const Napi::Value &value,
                ParsedImage *parsed) {
  if (!value.IsObject() || value.IsArray()) {
    ThrowTypeError(env, "image must be an object");
    return false;
  }
  Napi::Object object = value.As<Napi::Object>();
  if (!object.Has("data")) {
    ThrowTypeError(env, "image.data is required");
    return false;
  }
  Napi::Value data = object.Get("data");
  if (!data.IsBuffer() && !data.IsTypedArray()) {
    ThrowTypeError(env, "image.data must be a Buffer or Uint8Array");
    return false;
  }

  size_t byte_length = 0;
  uint8_t *data_ptr = nullptr;
  if (data.IsBuffer()) {
    auto buffer = data.As<Napi::Buffer<uint8_t>>();
    byte_length = buffer.Length();
    data_ptr = buffer.Data();
  } else {
    auto typed_array = data.As<Napi::TypedArray>();
    if (typed_array.TypedArrayType() != napi_uint8_array) {
      ThrowTypeError(env, "image.data must contain uint8 pixels");
      return false;
    }
    auto bytes = data.As<Napi::Uint8Array>();
    byte_length = bytes.ByteLength();
    data_ptr = bytes.Data();
  }

  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  if (!ReadInt(object, "width", &width) || width <= 0 ||
      !ReadInt(object, "height", &height) || height <= 0 ||
      !ReadInt(object, "channels", &channels) ||
      (channels != 3 && channels != 4)) {
    ThrowTypeError(env,
                   "image.width, image.height and image.channels (3 or 4) "
                   "are required");
    return false;
  }
  int32_t stride = 0;
  ReadInt(object, "stride", &stride);
  if (stride <= 0) stride = width * channels;
  if (stride < width * channels ||
      static_cast<uint64_t>(stride) * height > byte_length) {
    ThrowTypeError(env, "image.data is smaller than the declared image");
    return false;
  }
  int32_t format = ParseImageFormat(
      env, object.Has("format") ? object.Get("format") : env.Undefined());
  if (format < 0) return false;
  if ((format < 2 && channels != 3) || (format >= 2 && channels != 4)) {
    ThrowTypeError(env, "image.format does not match image.channels");
    return false;
  }

  parsed->data_holder = data;
  parsed->image.data = data_ptr;
  parsed->image.width = width;
  parsed->image.height = height;
  parsed->image.channels = channels;
  parsed->image.stride = stride;
  parsed->image.format = format;
  return true;
}

bool ReadNumberArray(const Napi::Env &env, const Napi::Value &value,
                     int32_t count, float *output, const char *name) {
  if (!value.IsArray()) {
    ThrowTypeError(env, std::string("face.") + name + " must be an array");
    return false;
  }
  Napi::Array array = value.As<Napi::Array>();
  if (array.Length() != static_cast<uint32_t>(count)) {
    std::ostringstream os;
    os << "face." << name << " must contain " << count << " numbers";
    ThrowTypeError(env, os.str());
    return false;
  }
  for (int32_t i = 0; i != count; ++i) {
    if (!array.Get(i).IsNumber()) {
      ThrowTypeError(env, std::string("face.") + name + " must contain numbers");
      return false;
    }
    output[i] = array.Get(i).As<Napi::Number>().FloatValue();
  }
  return true;
}

bool ParseFace(const Napi::Env &env, const Napi::Value &value,
               SherpaOnnxFaceDetection *face) {
  if (!value.IsObject() || value.IsArray()) {
    ThrowTypeError(env, "face must be an object");
    return false;
  }
  Napi::Object object = value.As<Napi::Object>();
  if (!object.Has("bbox")) {
    ThrowTypeError(env, "face.bbox is required");
    return false;
  }
  if (!ReadNumberArray(env, object.Get("bbox"), 4, face->bbox, "bbox")) {
    return false;
  }
  face->score = 0;
  ReadFloat(object, "score", &face->score);
  for (float &value : face->landmarks) {
    value = std::numeric_limits<float>::quiet_NaN();
  }
  if (object.Has("landmarks") &&
      !ReadNumberArray(env, object.Get("landmarks"), 10, face->landmarks,
                       "landmarks")) {
    return false;
  }
  return true;
}

Napi::Array MakeFaceArray(Napi::Env env,
                          const SherpaOnnxFaceDetectionResult *result) {
  Napi::Array array = Napi::Array::New(env, result->count);
  for (int32_t i = 0; i != result->count; ++i) {
    const SherpaOnnxFaceDetection &face = result->faces[i];
    Napi::Object object = Napi::Object::New(env);
    object.Set("index", Napi::Number::New(env, i));
    object.Set("score", Napi::Number::New(env, face.score));
    Napi::Array bbox = Napi::Array::New(env, 4);
    for (int32_t j = 0; j != 4; ++j) {
      bbox.Set(j, Napi::Number::New(env, face.bbox[j]));
    }
    Napi::Array landmarks = Napi::Array::New(env, 10);
    for (int32_t j = 0; j != 10; ++j) {
      landmarks.Set(j, Napi::Number::New(env, face.landmarks[j]));
    }
    object.Set("bbox", bbox);
    object.Set("landmarks", landmarks);
    array.Set(i, object);
  }
  return array;
}

Napi::External<SherpaOnnxRetinaFaceDetector>
CreateFaceDetectorWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsObject()) {
    ThrowTypeError(env, "createFaceDetector expects a config object");
    return {};
  }
  Napi::Object object = info[0].As<Napi::Object>();
  SherpaOnnxRetinaFaceConfig config{};
  std::string model = object.Has("model") && object.Get("model").IsString()
                          ? object.Get("model").As<Napi::String>().Utf8Value()
                          : "";
  std::string landmark_model =
      object.Has("landmarkModel") && object.Get("landmarkModel").IsString()
          ? object.Get("landmarkModel").As<Napi::String>().Utf8Value()
          : "";
  std::string provider =
      object.Has("provider") && object.Get("provider").IsString()
          ? object.Get("provider").As<Napi::String>().Utf8Value()
          : "cpu";
  config.model = model.c_str();
  config.landmark_model = landmark_model.c_str();
  config.provider = provider.c_str();
  ReadInt(object, "numThreads", &config.num_threads);
  ReadInt(object, "inputWidth", &config.input_width);
  ReadInt(object, "inputHeight", &config.input_height);
  ReadInt(object, "maxFaces", &config.max_faces);
  ReadFloat(object, "scoreThreshold", &config.score_threshold);
  ReadFloat(object, "nmsThreshold", &config.nms_threshold);
  if (object.Has("debug")) {
    if (object.Get("debug").IsBoolean()) {
      config.debug = object.Get("debug").As<Napi::Boolean>().Value();
    } else {
      ReadInt(object, "debug", &config.debug);
    }
  }
  const SherpaOnnxRetinaFaceDetector *detector =
      SherpaOnnxCreateRetinaFaceDetector(&config);
  if (!detector) {
    ThrowTypeError(env, "Failed to create ONNX face detector; check config and model");
    return {};
  }
  return Napi::External<SherpaOnnxRetinaFaceDetector>::New(
      env, const_cast<SherpaOnnxRetinaFaceDetector *>(detector),
      [](Napi::Env /*env*/, SherpaOnnxRetinaFaceDetector *value) {
        SherpaOnnxDestroyRetinaFaceDetector(value);
      });
}

Napi::Array FaceDetectorDetectWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsExternal()) {
    ThrowTypeError(env, "faceDetectorDetect expects a handle and image");
    return {};
  }
  ParsedImage parsed;
  if (!ParseImage(env, info[1], &parsed)) return {};
  auto detector = info[0]
                      .As<Napi::External<SherpaOnnxRetinaFaceDetector>>()
                      .Data();
  const SherpaOnnxFaceDetectionResult *result =
      SherpaOnnxRetinaFaceDetectorDetect(detector, &parsed.image);
  if (!result) {
    ThrowTypeError(env, "ONNX face detection failed");
    return {};
  }
  Napi::Array faces = MakeFaceArray(env, result);
  SherpaOnnxDestroyFaceDetectionResult(result);
  return faces;
}

// Async equivalent of FaceDetectorDetectWrapper. The image bytes are copied
// before the worker is queued because the JS-owned Buffer/Uint8Array may be
// changed or garbage-collected while ONNX Runtime is running.
class FaceDetectorDetectWorker : public Napi::AsyncWorker {
 public:
  FaceDetectorDetectWorker(
      Napi::Env env, Napi::Object handle,
      SherpaOnnxRetinaFaceDetector *detector, const SherpaOnnxImage &image,
      Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        model_handle_(Napi::Persistent(handle)),
        detector_(detector),
        image_(image),
        deferred_(deferred) {
    const size_t size = static_cast<size_t>(image.stride) * image.height;
    image_data_.resize(size);
    std::memcpy(image_data_.data(), image.data, size);
    image_.data = image_data_.data();
  }

  void Execute() override {
    try {
      result_ = SherpaOnnxRetinaFaceDetectorDetect(detector_, &image_);
      if (!result_) SetError("ONNX face detection failed");
    } catch (const std::exception &e) {
      SetError(e.what());
    } catch (...) {
      SetError("ONNX face detection failed");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    Napi::Array faces = MakeFaceArray(env, result_);
    SherpaOnnxDestroyFaceDetectionResult(result_);
    result_ = nullptr;
    deferred_.Resolve(faces);
    model_handle_.Reset();
  }

  void OnError(const Napi::Error &error) override {
    if (result_) {
      SherpaOnnxDestroyFaceDetectionResult(result_);
      result_ = nullptr;
    }
    deferred_.Reject(error.Value());
    model_handle_.Reset();
  }

 private:
  Napi::ObjectReference model_handle_;
  SherpaOnnxRetinaFaceDetector *detector_;
  SherpaOnnxImage image_{};
  std::vector<uint8_t> image_data_;
  Napi::Promise::Deferred deferred_;
  const SherpaOnnxFaceDetectionResult *result_ = nullptr;
};

Napi::Value FaceDetectorDetectAsyncWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsExternal()) {
    ThrowTypeError(env, "faceDetectorDetectAsync expects a handle and image");
    return env.Undefined();
  }
  ParsedImage parsed;
  if (!ParseImage(env, info[1], &parsed)) return env.Undefined();
  auto *detector = info[0]
                       .As<Napi::External<SherpaOnnxRetinaFaceDetector>>()
                       .Data();
  auto deferred = Napi::Promise::Deferred::New(env);
  auto *worker = new FaceDetectorDetectWorker(
      env, info[0].As<Napi::Object>(), detector, parsed.image, deferred);
  worker->Queue();
  return deferred.Promise();
}

Napi::External<SherpaOnnxAuraFaceRecognizer>
CreateAuraFaceRecognizerWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsObject()) {
    ThrowTypeError(env, "createAuraFaceRecognizer expects a config object");
    return {};
  }
  Napi::Object object = info[0].As<Napi::Object>();
  SherpaOnnxAuraFaceConfig config{};
  std::string model = object.Has("model") && object.Get("model").IsString()
                          ? object.Get("model").As<Napi::String>().Utf8Value()
                          : "";
  std::string provider =
      object.Has("provider") && object.Get("provider").IsString()
          ? object.Get("provider").As<Napi::String>().Utf8Value()
          : "cpu";
  config.model = model.c_str();
  config.provider = provider.c_str();
  ReadInt(object, "numThreads", &config.num_threads);
  ReadInt(object, "inputWidth", &config.input_width);
  ReadInt(object, "inputHeight", &config.input_height);
  if (object.Has("debug")) {
    if (object.Get("debug").IsBoolean()) {
      config.debug = object.Get("debug").As<Napi::Boolean>().Value();
    } else {
      ReadInt(object, "debug", &config.debug);
    }
  }
  const SherpaOnnxAuraFaceRecognizer *recognizer =
      SherpaOnnxCreateAuraFaceRecognizer(&config);
  if (!recognizer) {
    ThrowTypeError(env, "Failed to create AuraFace recognizer; check config and model");
    return {};
  }
  return Napi::External<SherpaOnnxAuraFaceRecognizer>::New(
      env, const_cast<SherpaOnnxAuraFaceRecognizer *>(recognizer),
      [](Napi::Env /*env*/, SherpaOnnxAuraFaceRecognizer *value) {
        SherpaOnnxDestroyAuraFaceRecognizer(value);
      });
}

Napi::Number AuraFaceRecognizerDimWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    ThrowTypeError(env, "auraFaceRecognizerDim expects a recognizer handle");
    return {};
  }
  auto recognizer = info[0]
                        .As<Napi::External<SherpaOnnxAuraFaceRecognizer>>()
                        .Data();
  return Napi::Number::New(env, SherpaOnnxAuraFaceRecognizerDim(recognizer));
}

Napi::Float32Array AuraFaceRecognizerComputeEmbeddingWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if ((info.Length() != 2 && info.Length() != 3) || !info[0].IsExternal()) {
    ThrowTypeError(env,
                   "auraFaceRecognizerComputeEmbedding expects handle, image "
                   "and optional face");
    return {};
  }
  ParsedImage parsed;
  if (!ParseImage(env, info[1], &parsed)) return {};
  SherpaOnnxFaceDetection face{};
  const SherpaOnnxFaceDetection *face_ptr = nullptr;
  if (info.Length() == 3 && !info[2].IsNull() && !info[2].IsUndefined()) {
    if (!ParseFace(env, info[2], &face)) return {};
    face_ptr = &face;
  }
  auto recognizer = info[0]
                        .As<Napi::External<SherpaOnnxAuraFaceRecognizer>>()
                        .Data();
  const float *embedding =
      SherpaOnnxAuraFaceRecognizerComputeEmbedding(recognizer, &parsed.image,
                                                   face_ptr);
  if (!embedding) {
    ThrowTypeError(env, "AuraFace embedding failed");
    return {};
  }
  int32_t dim = SherpaOnnxAuraFaceRecognizerDim(recognizer);
  Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(env, sizeof(float) * dim);
  Napi::Float32Array result = Napi::Float32Array::New(env, dim, buffer, 0);
  std::copy(embedding, embedding + dim, result.Data());
  SherpaOnnxAuraFaceRecognizerDestroyEmbedding(embedding);
  return result;
}

// Async AuraFace embedding worker. Like face detection, it owns a copy of
// the input pixels and optional detection so the JS values can be released
// while ONNX Runtime performs the embedding inference.
class AuraFaceEmbeddingWorker : public Napi::AsyncWorker {
 public:
  AuraFaceEmbeddingWorker(
      Napi::Env env, Napi::Object handle,
      SherpaOnnxAuraFaceRecognizer *recognizer, const SherpaOnnxImage &image,
      const SherpaOnnxFaceDetection *face, Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        model_handle_(Napi::Persistent(handle)),
        recognizer_(recognizer),
        image_(image),
        has_face_(face != nullptr),
        deferred_(deferred) {
    const size_t size = static_cast<size_t>(image.stride) * image.height;
    image_data_.resize(size);
    std::memcpy(image_data_.data(), image.data, size);
    image_.data = image_data_.data();
    if (face) face_ = *face;
  }

  void Execute() override {
    try {
      embedding_ = SherpaOnnxAuraFaceRecognizerComputeEmbedding(
          recognizer_, &image_, has_face_ ? &face_ : nullptr);
      if (!embedding_) SetError("AuraFace embedding failed");
    } catch (const std::exception &e) {
      SetError(e.what());
    } catch (...) {
      SetError("AuraFace embedding failed");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    const int32_t dim = SherpaOnnxAuraFaceRecognizerDim(recognizer_);
    Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(env, sizeof(float) * dim);
    Napi::Float32Array result = Napi::Float32Array::New(env, dim, buffer, 0);
    std::copy(embedding_, embedding_ + dim, result.Data());
    SherpaOnnxAuraFaceRecognizerDestroyEmbedding(embedding_);
    embedding_ = nullptr;
    deferred_.Resolve(result);
    model_handle_.Reset();
  }

  void OnError(const Napi::Error &error) override {
    if (embedding_) {
      SherpaOnnxAuraFaceRecognizerDestroyEmbedding(embedding_);
      embedding_ = nullptr;
    }
    deferred_.Reject(error.Value());
    model_handle_.Reset();
  }

 private:
  Napi::ObjectReference model_handle_;
  SherpaOnnxAuraFaceRecognizer *recognizer_;
  SherpaOnnxImage image_{};
  std::vector<uint8_t> image_data_;
  SherpaOnnxFaceDetection face_{};
  bool has_face_ = false;
  Napi::Promise::Deferred deferred_;
  const float *embedding_ = nullptr;
};

Napi::Value AuraFaceRecognizerComputeEmbeddingAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if ((info.Length() != 2 && info.Length() != 3) || !info[0].IsExternal()) {
    ThrowTypeError(env,
                   "auraFaceRecognizerComputeEmbeddingAsync expects handle, "
                   "image and optional face");
    return env.Undefined();
  }
  ParsedImage parsed;
  if (!ParseImage(env, info[1], &parsed)) return env.Undefined();
  SherpaOnnxFaceDetection face{};
  const SherpaOnnxFaceDetection *face_ptr = nullptr;
  if (info.Length() == 3 && !info[2].IsNull() && !info[2].IsUndefined()) {
    if (!ParseFace(env, info[2], &face)) return env.Undefined();
    face_ptr = &face;
  }
  auto *recognizer = info[0]
                         .As<Napi::External<SherpaOnnxAuraFaceRecognizer>>()
                         .Data();
  auto deferred = Napi::Promise::Deferred::New(env);
  auto *worker = new AuraFaceEmbeddingWorker(
      env, info[0].As<Napi::Object>(), recognizer, parsed.image, face_ptr,
      deferred);
  worker->Queue();
  return deferred.Promise();
}

Napi::Number FaceCosineSimilarityWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 && info.Length() != 3) {
    ThrowTypeError(env, "faceCosineSimilarity expects two embeddings and optional dimension");
    return {};
  }
  if (!info[0].IsTypedArray() || !info[1].IsTypedArray()) {
    ThrowTypeError(env, "embeddings must be Float32Array values");
    return {};
  }
  auto a = info[0].As<Napi::TypedArray>();
  auto b = info[1].As<Napi::TypedArray>();
  if (a.TypedArrayType() != napi_float32_array ||
      b.TypedArrayType() != napi_float32_array) {
    ThrowTypeError(env, "embeddings must be Float32Array values");
    return {};
  }
  auto a_values = info[0].As<Napi::Float32Array>();
  auto b_values = info[1].As<Napi::Float32Array>();
  int32_t dim = static_cast<int32_t>(std::min(a_values.ElementLength(),
                                              b_values.ElementLength()));
  if (info.Length() == 3) {
    if (!info[2].IsNumber()) {
      ThrowTypeError(env, "dimension must be a number");
      return {};
    }
    dim = info[2].As<Napi::Number>().Int32Value();
  }
  if (dim <= 0 || static_cast<size_t>(dim) > a_values.ElementLength() ||
      static_cast<size_t>(dim) > b_values.ElementLength()) {
    ThrowTypeError(env, "embedding dimension is out of range");
    return {};
  }
  return Napi::Number::New(
      env, SherpaOnnxFaceCosineSimilarity(a_values.Data(), b_values.Data(), dim));
}

}  // namespace

void InitFace(Napi::Env env, Napi::Object exports) {
  auto create_face_detector =
      Napi::Function::New(env, CreateFaceDetectorWrapper);
  auto face_detector_detect =
      Napi::Function::New(env, FaceDetectorDetectWrapper);
  auto face_detector_detect_async =
      Napi::Function::New(env, FaceDetectorDetectAsyncWrapper);
  exports.Set("createFaceDetector", create_face_detector);
  exports.Set("faceDetectorDetect", face_detector_detect);
  exports.Set("faceDetectorDetectAsync", face_detector_detect_async);
  // Keep the original native names for already-published JavaScript clients.
  exports.Set("createRetinaFaceDetector", create_face_detector);
  exports.Set("retinaFaceDetectorDetect", face_detector_detect);
  exports.Set("retinaFaceDetectorDetectAsync", face_detector_detect_async);
  exports.Set("createAuraFaceRecognizer",
              Napi::Function::New(env, CreateAuraFaceRecognizerWrapper));
  exports.Set("auraFaceRecognizerDim",
              Napi::Function::New(env, AuraFaceRecognizerDimWrapper));
  exports.Set("auraFaceRecognizerComputeEmbedding",
              Napi::Function::New(env, AuraFaceRecognizerComputeEmbeddingWrapper));
  exports.Set("auraFaceRecognizerComputeEmbeddingAsync",
              Napi::Function::New(env,
                                  AuraFaceRecognizerComputeEmbeddingAsyncWrapper));
  exports.Set("faceCosineSimilarity",
              Napi::Function::New(env, FaceCosineSimilarityWrapper));
}
