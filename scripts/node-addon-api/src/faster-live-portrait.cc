// Copyright (c) 2026 Xiaomi Corporation

#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

namespace {

void ThrowTypeError(const Napi::Env &env, const std::string &message) {
  Napi::TypeError::New(env, message).ThrowAsJavaScriptException();
}

bool ReadString(const Napi::Object &object, const char *name,
                std::string *value) {
  if (!object.Has(name) || !object.Get(name).IsString()) return false;
  *value = object.Get(name).As<Napi::String>().Utf8Value();
  return true;
}

bool ReadInt(const Napi::Object &object, const char *name, int32_t *value) {
  if (!object.Has(name) || !object.Get(name).IsNumber()) return false;
  *value = object.Get(name).As<Napi::Number>().Int32Value();
  return true;
}

bool ReadBool(const Napi::Object &object, const char *name, bool *value) {
  if (!object.Has(name) || !object.Get(name).IsBoolean()) return false;
  *value = object.Get(name).As<Napi::Boolean>().Value();
  return true;
}

bool AppendModel(const Napi::Env &env, const std::string &name,
                 const Napi::Value &value,
                 std::vector<std::string> *names,
                 std::vector<std::string> *paths) {
  std::string path;
  if (value.IsString()) {
    path = value.As<Napi::String>().Utf8Value();
  } else if (value.IsObject() && !value.IsArray()) {
    Napi::Object object = value.As<Napi::Object>();
    if (!ReadString(object, "path", &path)) {
      ReadString(object, "modelPath", &path);
    }
  }
  if (name.empty() || path.empty()) {
    ThrowTypeError(env, "Each FasterLivePortrait model needs a name and path");
    return false;
  }
  names->push_back(name);
  paths->push_back(std::move(path));
  return true;
}

bool ParseModels(const Napi::Env &env, const Napi::Object &config,
                 std::vector<std::string> *names,
                 std::vector<std::string> *paths) {
  if (!config.Has("models") || !config.Get("models").IsObject()) {
    ThrowTypeError(env, "FasterLivePortrait config.models must be an object or array");
    return false;
  }
  Napi::Value models_value = config.Get("models");
  if (models_value.IsArray()) {
    Napi::Array models = models_value.As<Napi::Array>();
    for (uint32_t i = 0; i != models.Length(); ++i) {
      if (!models.Get(i).IsObject() || models.Get(i).IsArray()) {
        ThrowTypeError(env, "FasterLivePortrait config.models array entries must be objects");
        return false;
      }
      Napi::Object item = models.Get(i).As<Napi::Object>();
      std::string name;
      std::string path;
      if (!ReadString(item, "name", &name) ||
          (!ReadString(item, "path", &path) &&
           !ReadString(item, "modelPath", &path))) {
        ThrowTypeError(env, "FasterLivePortrait model entries need name and path");
        return false;
      }
      names->push_back(std::move(name));
      paths->push_back(std::move(path));
    }
    if (names->empty()) {
      ThrowTypeError(env, "FasterLivePortrait config.models must not be empty");
      return false;
    }
    return true;
  }

  Napi::Object models = models_value.As<Napi::Object>();
  Napi::Array keys = models.GetPropertyNames();
  for (uint32_t i = 0; i != keys.Length(); ++i) {
    std::string name = keys.Get(i).As<Napi::String>().Utf8Value();
    if (!AppendModel(env, name, models.Get(name), names, paths)) return false;
  }
  if (names->empty()) {
    ThrowTypeError(env, "FasterLivePortrait config.models must not be empty");
    return false;
  }
  return true;
}

Napi::External<SherpaOnnxFasterLivePortraitModelSet>
CreateFasterLivePortraitModelSetWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsObject() || info[0].IsArray()) {
    ThrowTypeError(env, "createFasterLivePortraitModelSet expects a config object");
    return {};
  }

  Napi::Object object = info[0].As<Napi::Object>();
  std::vector<std::string> names;
  std::vector<std::string> paths;
  if (!ParseModels(env, object, &names, &paths)) return {};

  std::vector<SherpaOnnxFasterLivePortraitModel> models(names.size());
  for (size_t i = 0; i != models.size(); ++i) {
    models[i].name = names[i].c_str();
    models[i].path = paths[i].c_str();
  }
  std::string provider = "cpu";
  ReadString(object, "provider", &provider);
  int32_t num_threads = 1;
  ReadInt(object, "numThreads", &num_threads);
  bool debug = false;
  ReadBool(object, "debug", &debug);

  SherpaOnnxFasterLivePortraitConfig config{};
  config.models = models.data();
  config.num_models = static_cast<int32_t>(models.size());
  config.num_threads = num_threads;
  config.provider = provider.c_str();
  config.debug = debug ? 1 : 0;
  const auto *model_set = SherpaOnnxCreateFasterLivePortraitModelSet(&config);
  if (!model_set) {
    Napi::Error::New(env,
                     "Failed to create FasterLivePortrait model set; check model paths and provider")
        .ThrowAsJavaScriptException();
    return {};
  }
  return Napi::External<SherpaOnnxFasterLivePortraitModelSet>::New(
      env, const_cast<SherpaOnnxFasterLivePortraitModelSet *>(model_set),
      [](Napi::Env /*env*/, SherpaOnnxFasterLivePortraitModelSet *value) {
        SherpaOnnxDestroyFasterLivePortraitModelSet(value);
      });
}

class CreateFasterLivePortraitModelSetWorker : public Napi::AsyncWorker {
 public:
  CreateFasterLivePortraitModelSetWorker(
      Napi::Env env,
      std::vector<std::string> names,
      std::vector<std::string> paths,
      std::string provider,
      int32_t num_threads,
      bool debug,
      Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        names_(std::move(names)),
        paths_(std::move(paths)),
        provider_(std::move(provider)),
        num_threads_(num_threads),
        debug_(debug),
        deferred_(deferred) {}

  void Execute() override {
    try {
      std::vector<SherpaOnnxFasterLivePortraitModel> models(names_.size());
      for (size_t i = 0; i != models.size(); ++i) {
        models[i].name = names_[i].c_str();
        models[i].path = paths_[i].c_str();
      }
      SherpaOnnxFasterLivePortraitConfig config{};
      config.models = models.data();
      config.num_models = static_cast<int32_t>(models.size());
      config.num_threads = num_threads_;
      config.provider = provider_.c_str();
      config.debug = debug_ ? 1 : 0;
      model_set_ = SherpaOnnxCreateFasterLivePortraitModelSet(&config);
      if (!model_set_) {
        SetError("Failed to create FasterLivePortrait model set; check model paths and provider");
      }
    } catch (const std::exception &e) {
      SetError(e.what());
    } catch (...) {
      SetError("Failed to create FasterLivePortrait model set");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    auto external = Napi::External<SherpaOnnxFasterLivePortraitModelSet>::New(
        env, const_cast<SherpaOnnxFasterLivePortraitModelSet *>(model_set_),
        [](Napi::Env /*env*/, SherpaOnnxFasterLivePortraitModelSet *value) {
          SherpaOnnxDestroyFasterLivePortraitModelSet(value);
        });
    deferred_.Resolve(external);
  }

  void OnError(const Napi::Error &error) override {
    deferred_.Reject(error.Value());
  }

 private:
  std::vector<std::string> names_;
  std::vector<std::string> paths_;
  std::string provider_;
  int32_t num_threads_ = 1;
  bool debug_ = false;
  Napi::Promise::Deferred deferred_;
  const SherpaOnnxFasterLivePortraitModelSet *model_set_ = nullptr;
};

Napi::Value CreateFasterLivePortraitModelSetAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsObject()) {
    ThrowTypeError(env,
                   "createFasterLivePortraitModelSetAsync expects a config object");
    return env.Null();
  }
  Napi::Object object = info[0].As<Napi::Object>();
  std::vector<std::string> names;
  std::vector<std::string> paths;
  if (!ParseModels(env, object, &names, &paths)) return env.Null();

  std::string provider = "cpu";
  ReadString(object, "provider", &provider);
  int32_t num_threads = 1;
  ReadInt(object, "numThreads", &num_threads);
  bool debug = false;
  ReadBool(object, "debug", &debug);

  auto deferred = Napi::Promise::Deferred::New(env);
  auto *worker = new CreateFasterLivePortraitModelSetWorker(
      env, std::move(names), std::move(paths), std::move(provider),
      num_threads, debug, deferred);
  worker->Queue();
  return deferred.Promise();
}

Napi::String FasterLivePortraitGetModelInfoWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    ThrowTypeError(env, "fasterLivePortraitGetModelInfo expects a model-set handle");
    return {};
  }
  auto *model_set =
      info[0].As<Napi::External<SherpaOnnxFasterLivePortraitModelSet>>().Data();
  const char *json = SherpaOnnxFasterLivePortraitGetModelInfoJson(model_set);
  if (!json) {
    Napi::Error::New(env, "Failed to inspect FasterLivePortrait model set")
        .ThrowAsJavaScriptException();
    return {};
  }
  Napi::String result = Napi::String::New(env, json);
  SherpaOnnxFasterLivePortraitFreeString(json);
  return result;
}

struct ParsedTensor {
  Napi::Value data_holder;
  std::string name;
  std::vector<int64_t> shape;
  SherpaOnnxFasterLivePortraitTensor tensor{};
};

bool ParseShape(const Napi::Env &env, const Napi::Value &value,
                std::vector<int64_t> *shape) {
  if (!value.IsArray()) {
    ThrowTypeError(env, "FasterLivePortrait tensor.shape must be an array");
    return false;
  }
  Napi::Array array = value.As<Napi::Array>();
  shape->reserve(array.Length());
  for (uint32_t i = 0; i != array.Length(); ++i) {
    if (!array.Get(i).IsNumber()) {
      ThrowTypeError(env, "FasterLivePortrait tensor.shape must contain numbers");
      return false;
    }
    double value_number = array.Get(i).As<Napi::Number>().DoubleValue();
    int64_t dimension = static_cast<int64_t>(value_number);
    if (value_number != static_cast<double>(dimension) || dimension <= 0) {
      ThrowTypeError(env, "FasterLivePortrait tensor dimensions must be positive integers");
      return false;
    }
    shape->push_back(dimension);
  }
  return true;
}

bool ParseTensorType(const Napi::Env &env, const Napi::Object &object,
                     int32_t inferred_type, int32_t *type) {
  *type = inferred_type;
  if (!object.Has("type")) return true;
  if (!object.Get("type").IsString()) {
    ThrowTypeError(env, "FasterLivePortrait tensor.type must be a string");
    return false;
  }
  const std::string value = object.Get("type").As<Napi::String>().Utf8Value();
  if (value == "float32") *type = 1;
  else if (value == "float16") *type = 2;
  else if (value == "int64") *type = 3;
  else if (value == "int32") *type = 4;
  else if (value == "uint8") *type = 5;
  else if (value == "bool") *type = 6;
  else {
    ThrowTypeError(env, "Unsupported FasterLivePortrait tensor.type");
    return false;
  }
  return true;
}

bool ParseTensor(const Napi::Env &env, const Napi::Value &value,
                 ParsedTensor *parsed) {
  if (!value.IsObject() || value.IsArray()) {
    ThrowTypeError(env, "FasterLivePortrait inputs must be tensor objects");
    return false;
  }
  Napi::Object object = value.As<Napi::Object>();
  if (!object.Has("data") || !object.Get("data").IsTypedArray()) {
    ThrowTypeError(env,
                   "FasterLivePortrait tensor.data must be a typed array");
    return false;
  }
  if (!object.Has("shape") ||
      !ParseShape(env, object.Get("shape"), &parsed->shape)) {
    return false;
  }
  if (object.Has("name")) {
    if (!object.Get("name").IsString()) {
      ThrowTypeError(env, "FasterLivePortrait tensor.name must be a string");
      return false;
    }
    parsed->name = object.Get("name").As<Napi::String>().Utf8Value();
  }

  Napi::TypedArray data = object.Get("data").As<Napi::TypedArray>();
  int32_t inferred_type = 0;
  void *data_ptr = nullptr;
  size_t element_length = 0;
  switch (data.TypedArrayType()) {
    case napi_float32_array: {
      auto typed = object.Get("data").As<Napi::Float32Array>();
      inferred_type = 1;
      data_ptr = typed.Data();
      element_length = typed.ElementLength();
      break;
    }
    case napi_uint16_array: {
      auto typed = object.Get("data").As<Napi::Uint16Array>();
      inferred_type = 2;
      data_ptr = typed.Data();
      element_length = typed.ElementLength();
      break;
    }
    case napi_bigint64_array: {
      auto typed = object.Get("data").As<Napi::BigInt64Array>();
      inferred_type = 3;
      data_ptr = typed.Data();
      element_length = typed.ElementLength();
      break;
    }
    case napi_int32_array: {
      auto typed = object.Get("data").As<Napi::Int32Array>();
      inferred_type = 4;
      data_ptr = typed.Data();
      element_length = typed.ElementLength();
      break;
    }
    case napi_uint8_array: {
      auto typed = object.Get("data").As<Napi::Uint8Array>();
      inferred_type = 5;
      data_ptr = typed.Data();
      element_length = typed.ElementLength();
      break;
    }
    default:
      ThrowTypeError(env,
                     "FasterLivePortrait tensor.data supports Float32Array, Uint16Array, Int64Array, Int32Array and Uint8Array");
      return false;
  }
  if (!ParseTensorType(env, object, inferred_type, &parsed->tensor.type)) {
    return false;
  }
  if (parsed->tensor.type == 6) {
    if (inferred_type != 5) {
      ThrowTypeError(env, "bool tensors must use Uint8Array data");
      return false;
    }
  } else if (parsed->tensor.type != inferred_type) {
    ThrowTypeError(env,
                   "tensor.type does not match the supplied typed array; use Uint16Array for float16");
    return false;
  }

  size_t count = 1;
  for (int64_t dimension : parsed->shape) {
    if (static_cast<uint64_t>(dimension) >
        static_cast<uint64_t>(SIZE_MAX) / count) {
      ThrowTypeError(env, "FasterLivePortrait tensor is too large");
      return false;
    }
    count *= static_cast<size_t>(dimension);
  }
  if (count != element_length) {
    ThrowTypeError(env, "FasterLivePortrait tensor.data length does not match shape");
    return false;
  }
  parsed->data_holder = object.Get("data");
  parsed->tensor.name = parsed->name.empty() ? nullptr : parsed->name.c_str();
  parsed->tensor.shape = parsed->shape.data();
  parsed->tensor.rank = static_cast<int32_t>(parsed->shape.size());
  parsed->tensor.data = data_ptr;
  parsed->tensor.element_count = static_cast<int64_t>(count);
  return true;
}

Napi::Value MakeOutputTensor(const Napi::Env &env,
                             const SherpaOnnxFasterLivePortraitTensor *tensor) {
  Napi::Object result = Napi::Object::New(env);
  result.Set("name", Napi::String::New(env, tensor->name ? tensor->name : ""));
  const char *type_name = "";
  if (tensor->type == 1) type_name = "float32";
  else if (tensor->type == 2) type_name = "float16";
  else if (tensor->type == 3) type_name = "int64";
  else if (tensor->type == 4) type_name = "int32";
  else if (tensor->type == 5) type_name = "uint8";
  else if (tensor->type == 6) type_name = "bool";
  result.Set("type", Napi::String::New(env, type_name));
  Napi::Array shape = Napi::Array::New(env, tensor->rank);
  for (int32_t i = 0; i != tensor->rank; ++i) {
    shape.Set(i, Napi::Number::New(env, tensor->shape[i]));
  }
  result.Set("shape", shape);

  const size_t count = static_cast<size_t>(tensor->element_count);
  if (tensor->type == 1) {
    auto data = Napi::Float32Array::New(env, count);
    std::memcpy(data.Data(), tensor->data, count * sizeof(float));
    result.Set("data", data);
  } else if (tensor->type == 2) {
    auto data = Napi::Uint16Array::New(env, count);
    std::memcpy(data.Data(), tensor->data, count * sizeof(uint16_t));
    result.Set("data", data);
  } else if (tensor->type == 3) {
    auto data = Napi::BigInt64Array::New(env, count);
    std::memcpy(data.Data(), tensor->data, count * sizeof(int64_t));
    result.Set("data", data);
  } else if (tensor->type == 4) {
    auto data = Napi::Int32Array::New(env, count);
    std::memcpy(data.Data(), tensor->data, count * sizeof(int32_t));
    result.Set("data", data);
  } else {
    auto data = Napi::Uint8Array::New(env, count);
    std::memcpy(data.Data(), tensor->data, count * sizeof(uint8_t));
    result.Set("data", data);
  }
  return result;
}

Napi::Array FasterLivePortraitRunWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3 || !info[0].IsExternal() ||
      !info[1].IsString() || !info[2].IsArray()) {
    ThrowTypeError(env,
                   "fasterLivePortraitRun expects (modelSet, modelName, inputs[])");
    return {};
  }
  auto *model_set =
      info[0].As<Napi::External<SherpaOnnxFasterLivePortraitModelSet>>().Data();
  const std::string model_name = info[1].As<Napi::String>().Utf8Value();
  Napi::Array input_array = info[2].As<Napi::Array>();
  std::vector<ParsedTensor> parsed(input_array.Length());
  std::vector<SherpaOnnxFasterLivePortraitTensor> inputs(parsed.size());
  for (uint32_t i = 0; i != input_array.Length(); ++i) {
    if (!ParseTensor(env, input_array.Get(i), &parsed[i])) return {};
    inputs[i] = parsed[i].tensor;
  }
  const auto *result = SherpaOnnxFasterLivePortraitRun(
      model_set, model_name.c_str(), inputs.data(),
      static_cast<int32_t>(inputs.size()));
  if (!result) {
    Napi::Error::New(env, "FasterLivePortrait ONNX model run failed")
        .ThrowAsJavaScriptException();
    return {};
  }
  const int32_t count = SherpaOnnxFasterLivePortraitResultGetCount(result);
  Napi::Array outputs = Napi::Array::New(env, count);
  for (int32_t i = 0; i != count; ++i) {
    const auto *tensor =
        SherpaOnnxFasterLivePortraitResultGetTensor(result, i);
    outputs.Set(i, MakeOutputTensor(env, tensor));
  }
  SherpaOnnxDestroyFasterLivePortraitResult(result);
  return outputs;
}

// Async equivalent of FasterLivePortraitRunWrapper.  The synchronous method
// remains available for low-level callers, but the talking-video pipeline
// must use this worker so ORT inference never runs on Electron's JS thread.
class FasterLivePortraitRunWorker : public Napi::AsyncWorker {
 public:
  struct Input {
    std::string name;
    int32_t type = 0;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
  };

  FasterLivePortraitRunWorker(
      Napi::Env env, Napi::Object handle,
      SherpaOnnxFasterLivePortraitModelSet *model_set,
      std::string model_name, std::vector<Input> inputs,
      Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        model_handle_(Napi::Persistent(handle)),
        model_set_(model_set),
        model_name_(std::move(model_name)),
        inputs_(std::move(inputs)),
        deferred_(deferred) {}

  void Execute() override {
    try {
      std::vector<SherpaOnnxFasterLivePortraitTensor> tensors;
      tensors.reserve(inputs_.size());
      for (auto &input : inputs_) {
        SherpaOnnxFasterLivePortraitTensor tensor{};
        tensor.name = input.name.empty() ? nullptr : input.name.c_str();
        tensor.type = input.type;
        tensor.shape = input.shape.data();
        tensor.rank = static_cast<int32_t>(input.shape.size());
        tensor.data = input.data.data();
        tensor.element_count = static_cast<int64_t>(input.data.size());
        const size_t element_size = input.type == 1 ? 4 :
            (input.type == 2 ? 2 : (input.type == 3 ? 8 : 1));
        tensor.element_count = static_cast<int64_t>(input.data.size() / element_size);
        tensors.push_back(tensor);
      }
      result_ = SherpaOnnxFasterLivePortraitRun(
          model_set_, model_name_.c_str(), tensors.data(),
          static_cast<int32_t>(tensors.size()));
      if (!result_) SetError("FasterLivePortrait ONNX model run failed");
    } catch (const std::exception &e) {
      SetError(e.what());
    } catch (...) {
      SetError("FasterLivePortrait ONNX model run failed");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    Napi::Array outputs = Napi::Array::New(env);
    const int32_t count = SherpaOnnxFasterLivePortraitResultGetCount(result_);
    outputs = Napi::Array::New(env, count);
    for (int32_t i = 0; i != count; ++i) {
      outputs.Set(i, MakeOutputTensor(
          env, SherpaOnnxFasterLivePortraitResultGetTensor(result_, i)));
    }
    SherpaOnnxDestroyFasterLivePortraitResult(result_);
    result_ = nullptr;
    deferred_.Resolve(outputs);
    model_handle_.Reset();
  }

  void OnError(const Napi::Error &error) override {
    if (result_) {
      SherpaOnnxDestroyFasterLivePortraitResult(result_);
      result_ = nullptr;
    }
    deferred_.Reject(error.Value());
    model_handle_.Reset();
  }

 private:
  Napi::ObjectReference model_handle_;
  SherpaOnnxFasterLivePortraitModelSet *model_set_;
  std::string model_name_;
  std::vector<Input> inputs_;
  Napi::Promise::Deferred deferred_;
  const SherpaOnnxFasterLivePortraitResult *result_ = nullptr;
};

Napi::Value FasterLivePortraitRunAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 3 || !info[0].IsExternal() ||
      !info[1].IsString() || !info[2].IsArray()) {
    ThrowTypeError(env,
                   "fasterLivePortraitRunAsync expects (modelSet, modelName, inputs[])");
    return env.Undefined();
  }
  auto *model_set =
      info[0].As<Napi::External<SherpaOnnxFasterLivePortraitModelSet>>().Data();
  const std::string model_name = info[1].As<Napi::String>().Utf8Value();
  Napi::Array input_array = info[2].As<Napi::Array>();
  std::vector<FasterLivePortraitRunWorker::Input> inputs(input_array.Length());
  for (uint32_t i = 0; i != input_array.Length(); ++i) {
    ParsedTensor parsed;
    if (!ParseTensor(env, input_array.Get(i), &parsed)) return env.Undefined();
    auto &input = inputs[i];
    input.name = parsed.name;
    input.type = parsed.tensor.type;
    input.shape = std::move(parsed.shape);
    const size_t element_size = input.type == 1 ? 4 :
        (input.type == 2 ? 2 : (input.type == 3 ? 8 : 1));
    input.data.resize(static_cast<size_t>(parsed.tensor.element_count) * element_size);
    std::memcpy(input.data.data(), parsed.tensor.data, input.data.size());
  }

  auto deferred = Napi::Promise::Deferred::New(env);
  auto *worker = new FasterLivePortraitRunWorker(
      env, info[0].As<Napi::Object>(), model_set, model_name,
      std::move(inputs), deferred);
  worker->Queue();
  return deferred.Promise();
}

}  // namespace

void InitFasterLivePortrait(Napi::Env env, Napi::Object exports) {
  exports.Set(
      "createFasterLivePortraitModelSet",
      Napi::Function::New(env, CreateFasterLivePortraitModelSetWrapper));
  exports.Set(
      "createFasterLivePortraitModelSetAsync",
      Napi::Function::New(env, CreateFasterLivePortraitModelSetAsyncWrapper));
  exports.Set("fasterLivePortraitGetModelInfo",
              Napi::Function::New(env, FasterLivePortraitGetModelInfoWrapper));
  exports.Set("fasterLivePortraitRun",
              Napi::Function::New(env, FasterLivePortraitRunWrapper));
  exports.Set("fasterLivePortraitRunAsync",
              Napi::Function::New(env, FasterLivePortraitRunAsyncWrapper));
}
