// scripts/node-addon-api/src/streaming-speech-denoiser.cc
//
// Copyright (c)  2026  Xiaomi Corporation
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "napi.h"  // NOLINT
#include "sherpa-onnx/c-api/c-api.h"
#include "speech-denoiser.h"  // NOLINT

namespace {

struct OnlineDenoiserModelSettings {
  std::string gtcrn_model;
  std::string dpdfnet_model;
  std::string provider;
  int32_t num_threads = 0;
  int32_t debug = 0;
  float attenuation_limit_db = 0.0f;
};

OnlineDenoiserModelSettings ReadOnlineDenoiserModelSettings(
    Napi::Object object) {
  SherpaOnnxOfflineSpeechDenoiserModelConfig config =
      GetSpeechDenoiserModelConfig(object);
  OnlineDenoiserModelSettings settings;
  if (config.gtcrn.model) settings.gtcrn_model = config.gtcrn.model;
  if (config.dpdfnet.model) settings.dpdfnet_model = config.dpdfnet.model;
  if (config.provider) settings.provider = config.provider;
  settings.num_threads = config.num_threads;
  settings.debug = config.debug;
  settings.attenuation_limit_db = config.dpdfnet.attenuation_limit_db;
  DeleteSpeechDenoiserModelConfig(config);
  return settings;
}

Napi::Object CreateOwnedDenoisedAudioObject(Napi::Env env,
                                             const std::vector<float> &samples,
                                             int32_t sample_rate) {
  Napi::Object result = Napi::Object::New(env);
  Napi::ArrayBuffer buffer =
      Napi::ArrayBuffer::New(env, sizeof(float) * samples.size());
  Napi::Float32Array output =
      Napi::Float32Array::New(env, samples.size(), buffer, 0);
  if (!samples.empty()) {
    std::copy(samples.begin(), samples.end(), output.Data());
  }
  result.Set("samples", output);
  result.Set("sampleRate", sample_rate);
  return result;
}

void ThrowTypeError(Napi::Env env, const std::string &message) {
  Napi::TypeError::New(env, message).ThrowAsJavaScriptException();
}

}  // namespace

static Napi::External<SherpaOnnxOnlineSpeechDenoiser>
CreateOnlineSpeechDenoiserWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
#if __OHOS__
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();
    return {};
  }
#else
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();
    return {};
  }
#endif

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "Expect an object as the argument")
        .ThrowAsJavaScriptException();
    return {};
  }

  SherpaOnnxOnlineSpeechDenoiserConfig c;
  memset(&c, 0, sizeof(c));
  c.model = GetSpeechDenoiserModelConfig(info[0].As<Napi::Object>());

#if __OHOS__
  std::unique_ptr<NativeResourceManager,
                  decltype(&OH_ResourceManager_ReleaseNativeResourceManager)>
      mgr(OH_ResourceManager_InitNativeResourceManager(env, info[1]),
          &OH_ResourceManager_ReleaseNativeResourceManager);

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      SherpaOnnxCreateOnlineSpeechDenoiserOHOS(&c, mgr.get());
#else
  const SherpaOnnxOnlineSpeechDenoiser *sd =
      SherpaOnnxCreateOnlineSpeechDenoiser(&c);
#endif

  DeleteSpeechDenoiserModelConfig(c.model);

  if (!sd) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();
    return {};
  }

  return Napi::External<SherpaOnnxOnlineSpeechDenoiser>::New(
      env, const_cast<SherpaOnnxOnlineSpeechDenoiser *>(sd),
      [](Napi::Env /*env*/, SherpaOnnxOnlineSpeechDenoiser *sd) {
        SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
      });
}

class CreateOnlineSpeechDenoiserWorker : public Napi::AsyncWorker {
 public:
#if __OHOS__
  CreateOnlineSpeechDenoiserWorker(
      Napi::Env env, OnlineDenoiserModelSettings settings,
      NativeResourceManager *resource_manager, Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        settings_(std::move(settings)),
        resource_manager_(resource_manager),
        deferred_(deferred) {}
#else
  CreateOnlineSpeechDenoiserWorker(
      Napi::Env env, OnlineDenoiserModelSettings settings,
      Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        settings_(std::move(settings)),
        deferred_(deferred) {}
#endif

  ~CreateOnlineSpeechDenoiserWorker() override {
#if __OHOS__
    if (resource_manager_) {
      OH_ResourceManager_ReleaseNativeResourceManager(resource_manager_);
    }
#endif
  }

  Napi::Promise Promise() { return deferred_.Promise(); }

  void Execute() override {
    SherpaOnnxOnlineSpeechDenoiserConfig config{};
    config.model.gtcrn.model = settings_.gtcrn_model.empty()
                                   ? nullptr
                                   : settings_.gtcrn_model.c_str();
    config.model.dpdfnet.model = settings_.dpdfnet_model.empty()
                                     ? nullptr
                                     : settings_.dpdfnet_model.c_str();
    config.model.dpdfnet.attenuation_limit_db =
        settings_.attenuation_limit_db;
    config.model.provider =
        settings_.provider.empty() ? nullptr : settings_.provider.c_str();
    config.model.num_threads = settings_.num_threads;
    config.model.debug = settings_.debug;
#if __OHOS__
    if (resource_manager_) {
      denoiser_ = SherpaOnnxCreateOnlineSpeechDenoiserOHOS(
          &config, resource_manager_);
    } else {
      denoiser_ = SherpaOnnxCreateOnlineSpeechDenoiser(&config);
    }
#else
    denoiser_ = SherpaOnnxCreateOnlineSpeechDenoiser(&config);
#endif
    if (!denoiser_) SetError("Please check your config and model!");
  }

  void OnOK() override {
    Napi::Env env = Env();
    auto external = Napi::External<SherpaOnnxOnlineSpeechDenoiser>::New(
        env, const_cast<SherpaOnnxOnlineSpeechDenoiser *>(denoiser_),
        [](Napi::Env /*env*/, SherpaOnnxOnlineSpeechDenoiser *value) {
          SherpaOnnxDestroyOnlineSpeechDenoiser(value);
        });
    deferred_.Resolve(external);
  }

  void OnError(const Napi::Error &error) override {
    deferred_.Reject(error.Value());
  }

 private:
  OnlineDenoiserModelSettings settings_;
#if __OHOS__
  NativeResourceManager *resource_manager_ = nullptr;
#endif
  Napi::Promise::Deferred deferred_;
  const SherpaOnnxOnlineSpeechDenoiser *denoiser_ = nullptr;
};

static Napi::Value CreateOnlineSpeechDenoiserAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
#if __OHOS__
  if (info.Length() != 2) {
    ThrowTypeError(env, "createOnlineSpeechDenoiserAsync expects config and resources");
    return env.Undefined();
  }
#else
  if (info.Length() != 1) {
    ThrowTypeError(env, "createOnlineSpeechDenoiserAsync expects a config object");
    return env.Undefined();
  }
#endif
  if (!info[0].IsObject()) {
    ThrowTypeError(env, "Expect an object as the argument");
    return env.Undefined();
  }

  OnlineDenoiserModelSettings settings =
      ReadOnlineDenoiserModelSettings(info[0].As<Napi::Object>());
  auto deferred = Napi::Promise::Deferred::New(env);
#if __OHOS__
  auto *resource_manager =
      OH_ResourceManager_InitNativeResourceManager(env, info[1]);
  auto *worker = new CreateOnlineSpeechDenoiserWorker(
      env, std::move(settings), resource_manager, deferred);
#else
  auto *worker = new CreateOnlineSpeechDenoiserWorker(
      env, std::move(settings), deferred);
#endif
  worker->Queue();
  return deferred.Promise();
}

static Napi::Object OnlineSpeechDenoiserRunWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsExternal() || !info[1].IsObject()) {
    Napi::TypeError::New(env, "Expect a denoiser handle and an audio object")
        .ThrowAsJavaScriptException();
    return {};
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("samples") || !obj.Get("samples").IsTypedArray()) {
    Napi::TypeError::New(
        env, "The argument object should have a typed array field samples")
        .ThrowAsJavaScriptException();
    return {};
  }

  if (!obj.Has("sampleRate") || !obj.Get("sampleRate").IsNumber()) {
    Napi::TypeError::New(
        env, "The argument object should have a number field sampleRate")
        .ThrowAsJavaScriptException();
    return {};
  }

  Napi::Float32Array samples = obj.Get("samples").As<Napi::Float32Array>();
  int32_t sample_rate = obj.Get("sampleRate").As<Napi::Number>().Int32Value();
  const SherpaOnnxDenoisedAudio *audio = SherpaOnnxOnlineSpeechDenoiserRun(
      sd, samples.Data(), GetFloat32ArrayElementLength(samples), sample_rate);
  return CreateDenoisedAudioObject(env, audio, GetEnableExternalBuffer(obj));
}

static Napi::Object OnlineSpeechDenoiserFlushWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return {};
  }

  bool enable_external_buffer = true;
  if (info.Length() > 1 && info[1].IsBoolean()) {
    enable_external_buffer = info[1].As<Napi::Boolean>().Value();
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  const SherpaOnnxDenoisedAudio *audio =
      SherpaOnnxOnlineSpeechDenoiserFlush(sd);
  return CreateDenoisedAudioObject(env, audio, enable_external_buffer);
}

static void OnlineSpeechDenoiserResetWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return;
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  SherpaOnnxOnlineSpeechDenoiserReset(sd);
}

class OnlineSpeechDenoiserRunWorker : public Napi::AsyncWorker {
 public:
  OnlineSpeechDenoiserRunWorker(
      Napi::Env env, Napi::Object handle,
      const SherpaOnnxOnlineSpeechDenoiser *denoiser,
      std::vector<float> samples, int32_t sample_rate,
      Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        handle_(Napi::Persistent(handle)),
        denoiser_(denoiser),
        samples_(std::move(samples)),
        sample_rate_(sample_rate),
        output_sample_rate_(
            SherpaOnnxOnlineSpeechDenoiserGetSampleRate(denoiser)),
        deferred_(deferred) {}

  void Execute() override {
    try {
      const SherpaOnnxDenoisedAudio *audio = SherpaOnnxOnlineSpeechDenoiserRun(
          denoiser_, samples_.empty() ? nullptr : samples_.data(),
          static_cast<int32_t>(samples_.size()), sample_rate_);
      CopyOutput(audio);
    } catch (const std::exception &error) {
      SetError(error.what());
    } catch (...) {
      SetError("Online speech denoiser run failed");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    deferred_.Resolve(CreateOwnedDenoisedAudioObject(
        env, output_samples_, output_sample_rate_));
    handle_.Reset();
  }

  void OnError(const Napi::Error &error) override {
    deferred_.Reject(error.Value());
    handle_.Reset();
  }

 private:
  void CopyOutput(const SherpaOnnxDenoisedAudio *audio) {
    if (!audio) return;
    output_sample_rate_ = audio->sample_rate;
    if (audio->samples && audio->n > 0) {
      output_samples_.assign(audio->samples, audio->samples + audio->n);
    }
    SherpaOnnxDestroyDenoisedAudio(audio);
  }

  Napi::ObjectReference handle_;
  const SherpaOnnxOnlineSpeechDenoiser *denoiser_;
  std::vector<float> samples_;
  int32_t sample_rate_;
  int32_t output_sample_rate_;
  std::vector<float> output_samples_;
  Napi::Promise::Deferred deferred_;
};

static Napi::Value OnlineSpeechDenoiserRunAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2 || !info[0].IsExternal() || !info[1].IsObject()) {
    ThrowTypeError(env, "onlineSpeechDenoiserRunAsync expects a handle and audio object");
    return env.Undefined();
  }

  Napi::Object audio = info[1].As<Napi::Object>();
  if (!audio.Has("samples") || !audio.Get("samples").IsTypedArray() ||
      audio.Get("samples").As<Napi::TypedArray>().TypedArrayType() !=
          napi_float32_array) {
    ThrowTypeError(env, "audio.samples must be a Float32Array");
    return env.Undefined();
  }
  if (!audio.Has("sampleRate") || !audio.Get("sampleRate").IsNumber()) {
    ThrowTypeError(env, "audio.sampleRate must be a number");
    return env.Undefined();
  }

  Napi::Float32Array samples = audio.Get("samples").As<Napi::Float32Array>();
  std::vector<float> input_samples;
  const int32_t sample_count = GetFloat32ArrayElementLength(samples);
  if (sample_count > 0) {
    input_samples.assign(samples.Data(), samples.Data() + sample_count);
  }
  int32_t sample_rate = audio.Get("sampleRate").As<Napi::Number>().Int32Value();
  if (sample_rate <= 0) {
    ThrowTypeError(env, "audio.sampleRate must be greater than zero");
    return env.Undefined();
  }

  auto deferred = Napi::Promise::Deferred::New(env);
  auto *denoiser =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  auto *worker = new OnlineSpeechDenoiserRunWorker(
      env, info[0].As<Napi::Object>(), denoiser, std::move(input_samples),
      sample_rate, deferred);
  worker->Queue();
  return deferred.Promise();
}

class OnlineSpeechDenoiserFlushWorker : public Napi::AsyncWorker {
 public:
  OnlineSpeechDenoiserFlushWorker(
      Napi::Env env, Napi::Object handle,
      const SherpaOnnxOnlineSpeechDenoiser *denoiser,
      Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        handle_(Napi::Persistent(handle)),
        denoiser_(denoiser),
        output_sample_rate_(
            SherpaOnnxOnlineSpeechDenoiserGetSampleRate(denoiser)),
        deferred_(deferred) {}

  void Execute() override {
    try {
      const SherpaOnnxDenoisedAudio *audio =
          SherpaOnnxOnlineSpeechDenoiserFlush(denoiser_);
      if (audio) {
        output_sample_rate_ = audio->sample_rate;
        if (audio->samples && audio->n > 0) {
          output_samples_.assign(audio->samples, audio->samples + audio->n);
        }
        SherpaOnnxDestroyDenoisedAudio(audio);
      }
    } catch (const std::exception &error) {
      SetError(error.what());
    } catch (...) {
      SetError("Online speech denoiser flush failed");
    }
  }

  void OnOK() override {
    Napi::Env env = Env();
    deferred_.Resolve(CreateOwnedDenoisedAudioObject(
        env, output_samples_, output_sample_rate_));
    handle_.Reset();
  }

  void OnError(const Napi::Error &error) override {
    deferred_.Reject(error.Value());
    handle_.Reset();
  }

 private:
  Napi::ObjectReference handle_;
  const SherpaOnnxOnlineSpeechDenoiser *denoiser_;
  int32_t output_sample_rate_;
  std::vector<float> output_samples_;
  Napi::Promise::Deferred deferred_;
};

static Napi::Value OnlineSpeechDenoiserFlushAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || info.Length() > 2 || !info[0].IsExternal()) {
    ThrowTypeError(env, "onlineSpeechDenoiserFlushAsync expects a handle");
    return env.Undefined();
  }

  auto deferred = Napi::Promise::Deferred::New(env);
  auto *denoiser =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  auto *worker = new OnlineSpeechDenoiserFlushWorker(
      env, info[0].As<Napi::Object>(), denoiser, deferred);
  worker->Queue();
  return deferred.Promise();
}

class OnlineSpeechDenoiserResetWorker : public Napi::AsyncWorker {
 public:
  OnlineSpeechDenoiserResetWorker(
      Napi::Env env, Napi::Object handle,
      const SherpaOnnxOnlineSpeechDenoiser *denoiser,
      Napi::Promise::Deferred deferred)
      : Napi::AsyncWorker(env),
        handle_(Napi::Persistent(handle)),
        denoiser_(denoiser),
        deferred_(deferred) {}

  void Execute() override {
    try {
      SherpaOnnxOnlineSpeechDenoiserReset(denoiser_);
    } catch (const std::exception &error) {
      SetError(error.what());
    } catch (...) {
      SetError("Online speech denoiser reset failed");
    }
  }

  void OnOK() override {
    deferred_.Resolve(Env().Undefined());
    handle_.Reset();
  }

  void OnError(const Napi::Error &error) override {
    deferred_.Reject(error.Value());
    handle_.Reset();
  }

 private:
  Napi::ObjectReference handle_;
  const SherpaOnnxOnlineSpeechDenoiser *denoiser_;
  Napi::Promise::Deferred deferred_;
};

static Napi::Value OnlineSpeechDenoiserResetAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    ThrowTypeError(env, "onlineSpeechDenoiserResetAsync expects a handle");
    return env.Undefined();
  }

  auto deferred = Napi::Promise::Deferred::New(env);
  auto *denoiser =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  auto *worker = new OnlineSpeechDenoiserResetWorker(
      env, info[0].As<Napi::Object>(), denoiser, deferred);
  worker->Queue();
  return deferred.Promise();
}

static Napi::Number OnlineSpeechDenoiserGetSampleRateWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return {};
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  return Napi::Number::New(env,
                           SherpaOnnxOnlineSpeechDenoiserGetSampleRate(sd));
}

static Napi::Number OnlineSpeechDenoiserGetFrameShiftInSamplesWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expect an online speech denoiser pointer.")
        .ThrowAsJavaScriptException();
    return {};
  }

  const SherpaOnnxOnlineSpeechDenoiser *sd =
      info[0].As<Napi::External<SherpaOnnxOnlineSpeechDenoiser>>().Data();
  return Napi::Number::New(
      env, SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(sd));
}

void InitOnlineSpeechDenoiser(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOnlineSpeechDenoiser"),
              Napi::Function::New(env, CreateOnlineSpeechDenoiserWrapper));
  exports.Set(Napi::String::New(env, "createOnlineSpeechDenoiserAsync"),
              Napi::Function::New(env, CreateOnlineSpeechDenoiserAsyncWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserRunWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserRunWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserFlushWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserFlushWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserResetWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserResetWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserRunAsyncWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserRunAsyncWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserFlushAsyncWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserFlushAsyncWrapper));
  exports.Set(Napi::String::New(env, "onlineSpeechDenoiserResetAsyncWrapper"),
              Napi::Function::New(env, OnlineSpeechDenoiserResetAsyncWrapper));
  exports.Set(
      Napi::String::New(env, "onlineSpeechDenoiserGetSampleRateWrapper"),
      Napi::Function::New(env, OnlineSpeechDenoiserGetSampleRateWrapper));
  exports.Set(Napi::String::New(
                  env, "onlineSpeechDenoiserGetFrameShiftInSamplesWrapper"),
              Napi::Function::New(
                  env, OnlineSpeechDenoiserGetFrameShiftInSamplesWrapper));
}
