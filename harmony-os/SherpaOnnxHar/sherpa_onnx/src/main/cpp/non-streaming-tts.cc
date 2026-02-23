// scripts/node-addon-api/src/non-streaming-tts.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <algorithm>
#include <sstream>

#include "macros.h"  // NOLINT
#include "napi.h"    // NOLINT
#include "sherpa-onnx/c-api/c-api.h"

static SherpaOnnxOfflineTtsVitsModelConfig GetOfflineTtsVitsModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsVitsModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("vits") || !obj.Get("vits").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("vits").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  SHERPA_ONNX_ASSIGN_ATTR_STR(lexicon, lexicon);
  SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
  SHERPA_ONNX_ASSIGN_ATTR_STR(data_dir, dataDir);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(noise_scale, noiseScale);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(noise_scale_w, noiseScaleW);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(length_scale, lengthScale);

  return c;
}

static SherpaOnnxOfflineTtsMatchaModelConfig GetOfflineTtsMatchaModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsMatchaModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("matcha") || !obj.Get("matcha").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("matcha").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(acoustic_model, acousticModel);
  SHERPA_ONNX_ASSIGN_ATTR_STR(vocoder, vocoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(lexicon, lexicon);
  SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
  SHERPA_ONNX_ASSIGN_ATTR_STR(data_dir, dataDir);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(noise_scale, noiseScale);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(length_scale, lengthScale);

  return c;
}

static SherpaOnnxOfflineTtsKokoroModelConfig GetOfflineTtsKokoroModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsKokoroModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("kokoro") || !obj.Get("kokoro").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("kokoro").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  SHERPA_ONNX_ASSIGN_ATTR_STR(voices, voices);
  SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
  SHERPA_ONNX_ASSIGN_ATTR_STR(data_dir, dataDir);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(length_scale, lengthScale);
  SHERPA_ONNX_ASSIGN_ATTR_STR(lexicon, lexicon);
  SHERPA_ONNX_ASSIGN_ATTR_STR(g2p_model, g2p_model);
  SHERPA_ONNX_ASSIGN_ATTR_STR(lang, lang);

  return c;
}
static SherpaOnnxOfflineTtsChatterboxModelConfig GetOfflineTtsChatterboxModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsChatterboxModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("chatterbox") || !obj.Get("chatterbox").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("chatterbox").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(speech_encoder, speechEncoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(embed_tokens, embedTokens);
  SHERPA_ONNX_ASSIGN_ATTR_STR(language_model, languageModel);
  SHERPA_ONNX_ASSIGN_ATTR_STR(conditional_decoder, conditionalDecoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(tokenizer, tokenizer);
  SHERPA_ONNX_ASSIGN_ATTR_STR(lang, lang);
  SHERPA_ONNX_ASSIGN_ATTR_STR(lexicon, lexicon);
  SHERPA_ONNX_ASSIGN_ATTR_STR(cangjie_dict, cangjieDict);
  SHERPA_ONNX_ASSIGN_ATTR_STR(perth_watermarker, perthWatermarker);

  return c;
}
static SherpaOnnxOfflineTtsMiocodecLlamaModelConfig
GetOfflineTtsMiocodecLlamaModelConfig(Napi::Object obj) {
  SherpaOnnxOfflineTtsMiocodecLlamaModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("miocodecLlama") || !obj.Get("miocodecLlama").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("miocodecLlama").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  SHERPA_ONNX_ASSIGN_ATTR_STR(campplus_model, campplusModel);
  SHERPA_ONNX_ASSIGN_ATTR_STR(miocodec_encoder, miocodecEncoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(miocodec_decoder, miocodecDecoder);
  SHERPA_ONNX_ASSIGN_ATTR_STR(embeddings, embeddings);
  SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
  SHERPA_ONNX_ASSIGN_ATTR_STR(lexicon, lexicon);
  SHERPA_ONNX_ASSIGN_ATTR_STR(g2p_model, g2pModel);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(temperature, temperature);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(top_p, topP);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(max_tokens, maxTokens);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(repetition_penalty, repetitionPenalty);
  SHERPA_ONNX_ASSIGN_ATTR_STR(perth_watermarker, perthWatermarker);

  return c;
}
static SherpaOnnxOfflineTtsKittenModelConfig GetOfflineTtsKittenModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsKittenModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("kitten") || !obj.Get("kitten").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("kitten").As<Napi::Object>();
  SHERPA_ONNX_ASSIGN_ATTR_STR(model, model);
  SHERPA_ONNX_ASSIGN_ATTR_STR(voices, voices);
  SHERPA_ONNX_ASSIGN_ATTR_STR(tokens, tokens);
  SHERPA_ONNX_ASSIGN_ATTR_STR(data_dir, dataDir);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(length_scale, lengthScale);

  return c;
}

static SherpaOnnxOfflineTtsModelConfig GetOfflineTtsModelConfig(
    Napi::Object obj) {
  SherpaOnnxOfflineTtsModelConfig c;
  memset(&c, 0, sizeof(c));

  if (!obj.Has("model") || !obj.Get("model").IsObject()) {
    return c;
  }

  Napi::Object o = obj.Get("model").As<Napi::Object>();

  c.vits = GetOfflineTtsVitsModelConfig(o);
  c.matcha = GetOfflineTtsMatchaModelConfig(o);
  c.kokoro = GetOfflineTtsKokoroModelConfig(o);
  c.kitten = GetOfflineTtsKittenModelConfig(o);
  c.chatterbox = GetOfflineTtsChatterboxModelConfig(o);
  c.miocodec_llama = GetOfflineTtsMiocodecLlamaModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_INT32(num_threads, numThreads);

  if (o.Has("debug") &&
      (o.Get("debug").IsNumber() || o.Get("debug").IsBoolean())) {
    if (o.Get("debug").IsBoolean()) {
      c.debug = o.Get("debug").As<Napi::Boolean>().Value();
    } else {
      c.debug = o.Get("debug").As<Napi::Number>().Int32Value();
    }
  }

  SHERPA_ONNX_ASSIGN_ATTR_STR(provider, provider);

  return c;
}

static SherpaOnnxOfflineTtsConfig ParseTtsConfig(Napi::Object o) {
  SherpaOnnxOfflineTtsConfig c;
  memset(&c, 0, sizeof(c));

  c.model = GetOfflineTtsModelConfig(o);

  SHERPA_ONNX_ASSIGN_ATTR_STR(rule_fsts, ruleFsts);
  SHERPA_ONNX_ASSIGN_ATTR_INT32(max_num_sentences, maxNumSentences);
  SHERPA_ONNX_ASSIGN_ATTR_STR(rule_fars, ruleFars);
  SHERPA_ONNX_ASSIGN_ATTR_FLOAT(silence_scale, silenceScale);

  return c;
}

static void FreeTtsConfig(const SherpaOnnxOfflineTtsConfig &c) {
  SHERPA_ONNX_DELETE_C_STR(c.model.vits.model);
  SHERPA_ONNX_DELETE_C_STR(c.model.vits.lexicon);
  SHERPA_ONNX_DELETE_C_STR(c.model.vits.tokens);
  SHERPA_ONNX_DELETE_C_STR(c.model.vits.data_dir);

  SHERPA_ONNX_DELETE_C_STR(c.model.matcha.acoustic_model);
  SHERPA_ONNX_DELETE_C_STR(c.model.matcha.vocoder);
  SHERPA_ONNX_DELETE_C_STR(c.model.matcha.lexicon);
  SHERPA_ONNX_DELETE_C_STR(c.model.matcha.tokens);
  SHERPA_ONNX_DELETE_C_STR(c.model.matcha.data_dir);

  SHERPA_ONNX_DELETE_C_STR(c.model.kitten.model);
  SHERPA_ONNX_DELETE_C_STR(c.model.kitten.voices);
  SHERPA_ONNX_DELETE_C_STR(c.model.kitten.tokens);
  SHERPA_ONNX_DELETE_C_STR(c.model.kitten.data_dir);

  SHERPA_ONNX_DELETE_C_STR(c.model.kokoro.model);
  SHERPA_ONNX_DELETE_C_STR(c.model.kokoro.voices);
  SHERPA_ONNX_DELETE_C_STR(c.model.kokoro.tokens);
  SHERPA_ONNX_DELETE_C_STR(c.model.kokoro.lexicon);
  SHERPA_ONNX_DELETE_C_STR(c.model.kokoro.g2p_model);
  SHERPA_ONNX_DELETE_C_STR(c.model.kokoro.lang);

  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.speech_encoder);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.embed_tokens);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.language_model);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.conditional_decoder);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.tokenizer);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.lang);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.lexicon);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.cangjie_dict);
  SHERPA_ONNX_DELETE_C_STR(c.model.chatterbox.perth_watermarker);

  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.model);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.campplus_model);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.miocodec_encoder);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.miocodec_decoder);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.embeddings);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.tokens);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.lexicon);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.g2p_model);
  SHERPA_ONNX_DELETE_C_STR(c.model.miocodec_llama.perth_watermarker);

  SHERPA_ONNX_DELETE_C_STR(c.model.provider);

  SHERPA_ONNX_DELETE_C_STR(c.rule_fsts);
  SHERPA_ONNX_DELETE_C_STR(c.rule_fars);
}

static Napi::External<SherpaOnnxOfflineTts> CreateOfflineTtsWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
#if __OHOS__
  // the last argument is the NativeResourceManager
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

  Napi::Object o = info[0].As<Napi::Object>();
  SherpaOnnxOfflineTtsConfig c = ParseTtsConfig(o);

#if __OHOS__
  std::unique_ptr<NativeResourceManager,
                  decltype(&OH_ResourceManager_ReleaseNativeResourceManager)>
      mgr(OH_ResourceManager_InitNativeResourceManager(env, info[1]),
          &OH_ResourceManager_ReleaseNativeResourceManager);
  const SherpaOnnxOfflineTts *tts =
      SherpaOnnxCreateOfflineTtsOHOS(&c, mgr.get());
#else
  const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&c);
#endif
  FreeTtsConfig(c);

  if (!tts) {
    Napi::TypeError::New(env, "Please check your config!")
        .ThrowAsJavaScriptException();

    return {};
  }

  return Napi::External<SherpaOnnxOfflineTts>::New(
      env, const_cast<SherpaOnnxOfflineTts *>(tts),
      [](Napi::Env env, SherpaOnnxOfflineTts *tts) {
        SherpaOnnxDestroyOfflineTts(tts);
      });
}

class CreateOfflineTtsWorker : public Napi::AsyncWorker {
 public:
#if __OHOS__
  CreateOfflineTtsWorker(const Napi::Env &env,
                         const SherpaOnnxOfflineTtsConfig &config,
                         NativeResourceManager *mgr)
      : Napi::AsyncWorker(env), deferred_(env), config_(config), mgr_(mgr) {}
#else
  CreateOfflineTtsWorker(const Napi::Env &env,
                         const SherpaOnnxOfflineTtsConfig &config)
      : Napi::AsyncWorker(env), deferred_(env), config_(config) {}
#endif

  ~CreateOfflineTtsWorker() {
    FreeTtsConfig(config_);
#if __OHOS__
    if (mgr_) {
      OH_ResourceManager_ReleaseNativeResourceManager(mgr_);
    }
#endif
  }

  Napi::Promise Promise() { return deferred_.Promise(); }

 protected:
  void Execute() override {
#if __OHOS__
    if (mgr_) {
      tts_ = SherpaOnnxCreateOfflineTtsOHOS(&config_, mgr_);
    } else {
      tts_ = SherpaOnnxCreateOfflineTts(&config_);
    }
#else
    tts_ = SherpaOnnxCreateOfflineTts(&config_);
#endif
  }

  void OnOK() override {
    Napi::Env env = Env();
    if (!tts_) {
      deferred_.Reject(
          Napi::TypeError::New(env, "Please check your config!").Value());
      return;
    }

    auto external = Napi::External<SherpaOnnxOfflineTts>::New(
        env, const_cast<SherpaOnnxOfflineTts *>(tts_),
        [](Napi::Env env, SherpaOnnxOfflineTts *tts) {
          SherpaOnnxDestroyOfflineTts(tts);
        });

    deferred_.Resolve(external);
  }

 private:
  Napi::Promise::Deferred deferred_;
  SherpaOnnxOfflineTtsConfig config_;
  const SherpaOnnxOfflineTts *tts_ = nullptr;
#if __OHOS__
  NativeResourceManager *mgr_ = nullptr;
#endif
};

static Napi::Value CreateOfflineTtsAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

#if __OHOS__
  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return env.Null();
  }
#else
  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return env.Null();
  }
#endif

  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "Expect an object as the argument")
        .ThrowAsJavaScriptException();

    return env.Null();
  }

  Napi::Object o = info[0].As<Napi::Object>();
  SherpaOnnxOfflineTtsConfig c = ParseTtsConfig(o);

#if __OHOS__
  NativeResourceManager *mgr =
      OH_ResourceManager_InitNativeResourceManager(env, info[1]);

  CreateOfflineTtsWorker *worker = new CreateOfflineTtsWorker(env, c, mgr);
#else
  CreateOfflineTtsWorker *worker = new CreateOfflineTtsWorker(env, c);
#endif
  worker->Queue();

  return worker->Promise();
}

static Napi::Number OfflineTtsSampleRateWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineTts *tts =
      info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

  int32_t sample_rate = SherpaOnnxOfflineTtsSampleRate(tts);

  return Napi::Number::New(env, sample_rate);
}

static Napi::Number OfflineTtsNumSpeakersWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 1) {
    std::ostringstream os;
    os << "Expect only 1 argument. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineTts *tts =
      info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

  int32_t num_speakers = SherpaOnnxOfflineTtsNumSpeakers(tts);

  return Napi::Number::New(env, num_speakers);
}

// synchronous version
static Napi::Object OfflineTtsGenerateWrapper(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineTts *tts =
      info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("text")) {
    Napi::TypeError::New(env, "The argument object should have a field text")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("text").IsString()) {
    Napi::TypeError::New(env, "The object['text'] should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("sid")) {
    Napi::TypeError::New(env, "The argument object should have a field sid")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("sid").IsNumber()) {
    Napi::TypeError::New(env, "The object['sid'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("speed")) {
    Napi::TypeError::New(env, "The argument object should have a field speed")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("speed").IsNumber()) {
    Napi::TypeError::New(env, "The object['speed'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  bool enable_external_buffer = true;
  if (obj.Has("enableExternalBuffer") &&
      obj.Get("enableExternalBuffer").IsBoolean()) {
    enable_external_buffer =
        obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
  }

  Napi::String _text = obj.Get("text").As<Napi::String>();
  std::string text = _text.Utf8Value();
  int32_t sid = obj.Get("sid").As<Napi::Number>().Int32Value();
  float speed = obj.Get("speed").As<Napi::Number>().FloatValue();
  bool g2p = false;
  if (obj.Has("g2p") && obj.Get("g2p").IsBoolean()) {
    g2p = obj.Get("g2p").As<Napi::Boolean>().Value();
  }
  std::string lang;
  if (obj.Has("lang") && obj.Get("lang").IsString()) {
    Napi::String _lang = obj.Get("lang").As<Napi::String>();
    lang = _lang.Utf8Value();
  }
  const SherpaOnnxGeneratedAudio *audio = nullptr;
  if (obj.Has("audioDir") && obj.Get("audioDir").IsString()) {
    std::string audio_dir = obj.Get("audioDir").As<Napi::String>().Utf8Value();
    audio = SherpaOnnxOfflineTtsGenerateWithMiocodecLlama(tts, text.c_str(), audio_dir.c_str(), speed, lang.c_str(), nullptr, nullptr);
  } else {
    try {
      audio = SherpaOnnxOfflineTtsGenerate(tts, text.c_str(), sid, speed, g2p, lang.c_str());
    } catch (...) {
      audio = nullptr;
    }
  }

  if (!audio) {
    return {};
  }
  if (enable_external_buffer) {
    Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
        env, const_cast<float *>(audio->samples), sizeof(float) * audio->n,
        [](Napi::Env /*env*/, void * /*data*/,
           const SherpaOnnxGeneratedAudio *hint) {
          SherpaOnnxDestroyOfflineTtsGeneratedAudio(hint);
        },
        audio);
    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);

    Napi::Object ans = Napi::Object::New(env);
    ans.Set(Napi::String::New(env, "samples"), float32Array);
    ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
    return ans;
  } else {
    // don't use external buffer
    Napi::ArrayBuffer arrayBuffer =
        Napi::ArrayBuffer::New(env, sizeof(float) * audio->n);

    Napi::Float32Array float32Array =
        Napi::Float32Array::New(env, audio->n, arrayBuffer, 0);

    std::copy(audio->samples, audio->samples + audio->n, float32Array.Data());

    Napi::Object ans = Napi::Object::New(env);
    ans.Set(Napi::String::New(env, "samples"), float32Array);
    ans.Set(Napi::String::New(env, "sampleRate"), audio->sample_rate);
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    return ans;
  }
}

struct TtsCallbackData {
  std::vector<float> samples;
  float progress;
  bool processed = false;
  bool cancelled = false;
};

// see
// https://github.com/nodejs/node-addon-examples/blob/main/src/6-threadsafe-function/typed_threadsafe_function/node-addon-api/clock.cc
static void InvokeJsCallback(Napi::Env env, Napi::Function callback,
                             Napi::Reference<Napi::Value> *context,
                             TtsCallbackData *data) {
  if (env != nullptr) {
    if (callback != nullptr) {
      Napi::ArrayBuffer arrayBuffer =
          Napi::ArrayBuffer::New(env, sizeof(float) * data->samples.size());

      Napi::Float32Array float32Array =
          Napi::Float32Array::New(env, data->samples.size(), arrayBuffer, 0);

      std::copy(data->samples.begin(), data->samples.end(),
                float32Array.Data());

      Napi::Object arg = Napi::Object::New(env);
      arg.Set(Napi::String::New(env, "samples"), float32Array);
      arg.Set(Napi::String::New(env, "progress"), data->progress);

      auto v = callback.Call(context->Value(), {arg});
      data->processed = true;
      if (v.IsNumber() && v.As<Napi::Number>().Int32Value()) {
        data->cancelled = false;
      } else {
        data->cancelled = true;
      }
    }
  }
}

using TSFN = Napi::TypedThreadSafeFunction<Napi::Reference<Napi::Value>,
                                           TtsCallbackData, InvokeJsCallback>;

class TtsGenerateWorker : public Napi::AsyncWorker {
 public:
  TtsGenerateWorker(const Napi::Env &env, TSFN tsfn,
                    const SherpaOnnxOfflineTts *tts, const std::string &text,
                    float speed, int32_t sid, bool use_external_buffer, bool g2p,
                    const std::string &lang, const std::string &audio_dir, const std::string &type, float exaggeration)
      : tsfn_(tsfn),
        Napi::AsyncWorker{env, "TtsGenerateWorker"},
        deferred_(env),
        tts_(tts),
        text_(text),
        speed_(speed),
        sid_(sid),
        use_external_buffer_(use_external_buffer),
        g2p_(g2p),
        lang_(lang),
        audio_dir(audio_dir),
        type(type),
        exaggeration_(exaggeration) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

  ~TtsGenerateWorker() {
    for (auto d : data_list_) {
      delete d;
    }
  }

 protected:
  void Execute() override {
    auto callback = [](const float *samples, int32_t n, float progress,
                       void *arg) -> int32_t {
      TtsGenerateWorker *_this = reinterpret_cast<TtsGenerateWorker *>(arg);

      for (auto d : _this->data_list_) {
        if (d->cancelled) {
#if __OHOS__
          OH_LOG_INFO(LOG_APP, "TtsGenerate is cancelled");
#endif
          return 0;
        }
      }

      auto data = new TtsCallbackData;
      data->samples = std::vector<float>{samples, samples + n};
      data->progress = progress;
      _this->data_list_.push_back(data);

      _this->tsfn_.NonBlockingCall(data);

      return 1;
    };
    if(type == "chatterbox") {
      audio_ = SherpaOnnxOfflineTtsGenerateWithChatterbox(
        tts_, text_.c_str(), audio_dir.c_str(), speed_, lang_.c_str(), exaggeration_, callback, this);
    } else {
      audio_ = SherpaOnnxOfflineTtsGenerateWithProgressCallbackWithArg(
        tts_, text_.c_str(), sid_, speed_,g2p_, lang_.c_str(), callback, this);
    }

    tsfn_.Release();
  }

  void OnOK() override {
    Napi::Env env = deferred_.Env();
    Napi::Object ans = Napi::Object::New(env);
    if (use_external_buffer_) {
      Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(
          env, const_cast<float *>(audio_->samples), sizeof(float) * audio_->n,
          [](Napi::Env /*env*/, void * /*data*/,
             const SherpaOnnxGeneratedAudio *hint) {
            SherpaOnnxDestroyOfflineTtsGeneratedAudio(hint);
          },
          audio_);
      Napi::Float32Array float32Array =
          Napi::Float32Array::New(env, audio_->n, arrayBuffer, 0);

      ans.Set(Napi::String::New(env, "samples"), float32Array);
      ans.Set(Napi::String::New(env, "sampleRate"), audio_->sample_rate);
    } else {
      // don't use external buffer
      Napi::ArrayBuffer arrayBuffer =
          Napi::ArrayBuffer::New(env, sizeof(float) * audio_->n);

      Napi::Float32Array float32Array =
          Napi::Float32Array::New(env, audio_->n, arrayBuffer, 0);

      std::copy(audio_->samples, audio_->samples + audio_->n,
                float32Array.Data());

      ans.Set(Napi::String::New(env, "samples"), float32Array);
      ans.Set(Napi::String::New(env, "sampleRate"), audio_->sample_rate);
      SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_);
    }

    deferred_.Resolve(ans);
  }

 private:
  TSFN tsfn_;
  Napi::Promise::Deferred deferred_;
  const SherpaOnnxOfflineTts *tts_;
  std::string text_;
  float speed_;
  int32_t sid_;
  bool use_external_buffer_;
  bool g2p_;
  std::string lang_;
  std::string audio_dir;
  std::string type;
  float exaggeration_;

  const SherpaOnnxGeneratedAudio *audio_;

  std::vector<TtsCallbackData *> data_list_;
};

static Napi::Object OfflineTtsGenerateAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() != 2) {
    std::ostringstream os;
    os << "Expect only 2 arguments. Given: " << info.Length();

    Napi::TypeError::New(env, os.str()).ThrowAsJavaScriptException();

    return {};
  }

  if (!info[0].IsExternal()) {
    Napi::TypeError::New(env, "Argument 0 should be an offline tts pointer.")
        .ThrowAsJavaScriptException();

    return {};
  }

  const SherpaOnnxOfflineTts *tts =
      info[0].As<Napi::External<SherpaOnnxOfflineTts>>().Data();

  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "Argument 1 should be an object")
        .ThrowAsJavaScriptException();

    return {};
  }

  Napi::Object obj = info[1].As<Napi::Object>();

  if (!obj.Has("text")) {
    Napi::TypeError::New(env, "The argument object should have a field text")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("text").IsString()) {
    Napi::TypeError::New(env, "The object['text'] should be a string")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("sid")) {
    Napi::TypeError::New(env, "The argument object should have a field sid")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("sid").IsNumber()) {
    Napi::TypeError::New(env, "The object['sid'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Has("speed")) {
    Napi::TypeError::New(env, "The argument object should have a field speed")
        .ThrowAsJavaScriptException();

    return {};
  }

  if (!obj.Get("speed").IsNumber()) {
    Napi::TypeError::New(env, "The object['speed'] should be a number")
        .ThrowAsJavaScriptException();

    return {};
  }

  bool enable_external_buffer = true;
  if (obj.Has("enableExternalBuffer") &&
      obj.Get("enableExternalBuffer").IsBoolean()) {
    enable_external_buffer =
        obj.Get("enableExternalBuffer").As<Napi::Boolean>().Value();
  }

  Napi::String _text = obj.Get("text").As<Napi::String>();
  std::string text = _text.Utf8Value();
  int32_t sid = obj.Get("sid").As<Napi::Number>().Int32Value();
  float speed = obj.Get("speed").As<Napi::Number>().FloatValue();
  bool g2p = false;
  if (obj.Has("g2p") && obj.Get("g2p").IsBoolean()) {
    g2p = obj.Get("g2p").As<Napi::Boolean>().Value();
  }
  std::string lang;
  if (obj.Has("lang") && obj.Get("lang").IsString()) {
    Napi::String _lang = obj.Get("lang").As<Napi::String>();
    lang = _lang.Utf8Value();
  }
  Napi::Function cb;
  if (obj.Has("callback") && obj.Get("callback").IsFunction()) {
    cb = obj.Get("callback").As<Napi::Function>();
  }
  
  std::string audio_dir;
  if (obj.Has("audioDir") && obj.Get("audioDir").IsString()) {
    Napi::String _audio_dir = obj.Get("audioDir").As<Napi::String>();
    audio_dir = _audio_dir.Utf8Value();
  }
  std::string type = "normal";
  if (obj.Has("type") && obj.Get("type").IsString()) {
    Napi::String _type = obj.Get("type").As<Napi::String>();
    type = _type.Utf8Value();
  }
  float exaggeration = 0.5;
  if (obj.Has("exaggeration") && obj.Get("exaggeration").IsNumber()) {
    exaggeration = obj.Get("exaggeration").As<Napi::Number>().FloatValue();
  }
  
  auto context =
      new Napi::Reference<Napi::Value>(Napi::Persistent(info.This()));

  TSFN tsfn = TSFN::New(
      env,
      cb,                 // JavaScript function called asynchronously
      "TtsGenerateFunc",  // Name
      0,                  // Unlimited queue
      1,                  // Only one thread will use this initially
      context,
      [](Napi::Env, void *, Napi::Reference<Napi::Value> *ctx) { delete ctx; });

  TtsGenerateWorker *worker = new TtsGenerateWorker(
      env, tsfn, tts, text, speed, sid, enable_external_buffer, g2p, lang, audio_dir, type, exaggeration);
  worker->Queue();
  return worker->Promise();
}

static Napi::Object GetEmbeddingsObject(Napi::Env env, 
    const SherpaOnnxOfflineTtsMiocodecLlamaEmbeddings *emb) {
  Napi::Object obj = Napi::Object::New(env);
  
  if (emb) {
    // Speaker Embedding
    Napi::Float32Array spk = Napi::Float32Array::New(env, emb->speaker_embedding_dim);
    memcpy(spk.Data(), emb->speaker_embedding, emb->speaker_embedding_dim * sizeof(float));
    obj.Set("speakerEmbedding", spk);

    // Global Embedding
    Napi::Float32Array global = Napi::Float32Array::New(env, emb->global_embedding_dim);
    memcpy(global.Data(), emb->global_embedding, emb->global_embedding_dim * sizeof(float));
    obj.Set("globalEmbedding", global);
  }
  
  return obj;
}

static Napi::Value OfflineTtsExtractMiocodecLlamaEmbeddingsWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
    Napi::TypeError::New(env, "Expect 2 arguments: tts, audioDir").ThrowAsJavaScriptException();
    return env.Null();
  }

  SherpaOnnxOfflineTts *tts =
      reinterpret_cast<SherpaOnnxOfflineTts *>(info[0].As<Napi::External<void>>().Data());
  std::string audio_dir = info[1].As<Napi::String>().Utf8Value();

  const auto *emb = SherpaOnnxOfflineTtsMiocodecLlamaExtractEmbeddings(tts, audio_dir.c_str());
  if (!emb) {
    return env.Undefined();
  }
  
  Napi::Object result = GetEmbeddingsObject(env, emb);
  SherpaOnnxDestroyOfflineTtsMiocodecLlamaEmbeddings(emb);
  return result;
}

class OfflineTtsMiocodecLlamaExtractEmbeddingsWorker : public Napi::AsyncWorker {
 public:
  OfflineTtsMiocodecLlamaExtractEmbeddingsWorker(Napi::Env env,
                                                 const SherpaOnnxOfflineTts *tts,
                                                 std::string audio_dir)
      : Napi::AsyncWorker(env), tts_(tts), audio_dir_(std::move(audio_dir)), emb_(nullptr), deferred_(Napi::Promise::Deferred::New(env)) {}

  ~OfflineTtsMiocodecLlamaExtractEmbeddingsWorker() {
    if (emb_) {
      SherpaOnnxDestroyOfflineTtsMiocodecLlamaEmbeddings(emb_);
    }
  }

  void Execute() override {
    emb_ = SherpaOnnxOfflineTtsMiocodecLlamaExtractEmbeddings(tts_, audio_dir_.c_str());
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    if (emb_) {
      deferred_.Resolve(GetEmbeddingsObject(Env(), emb_));
    } else {
       deferred_.Resolve(Env().Undefined());
    }
  }
  
  void OnError(const Napi::Error& e) override {
      deferred_.Reject(Napi::String::New(Env(), e.Message()));
  }

  Napi::Promise Promise() { return deferred_.Promise(); }

 private:
  const SherpaOnnxOfflineTts *tts_;
  std::string audio_dir_;
  const SherpaOnnxOfflineTtsMiocodecLlamaEmbeddings *emb_;
  Napi::Promise::Deferred deferred_;
};

static Napi::Value OfflineTtsExtractMiocodecLlamaEmbeddingsAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() != 2) {
      // tts, audioDir
    Napi::TypeError::New(env, "Expect 2 arguments: tts, audioDir").ThrowAsJavaScriptException();
    return env.Null();
  }

  SherpaOnnxOfflineTts *tts =
      reinterpret_cast<SherpaOnnxOfflineTts *>(info[0].As<Napi::External<void>>().Data());
  std::string audio_dir = info[1].As<Napi::String>().Utf8Value();

  auto *worker = new OfflineTtsMiocodecLlamaExtractEmbeddingsWorker(env, tts, audio_dir);
  worker->Queue();
  return worker->Promise();
}

// --------------------------------------------------------------------------------

class OfflineTtsGenerateWithMiocodecLlamaEmbeddingsWorker : public Napi::AsyncWorker {
 public:
  OfflineTtsGenerateWithMiocodecLlamaEmbeddingsWorker(Napi::Env env,
                                                      const SherpaOnnxOfflineTts *tts,
                                                      std::string text,
                                                      std::vector<float> spk_emb,
                                                      std::vector<float> global_emb,
                                                      float speed,
                                                      std::string lang)
      : Napi::AsyncWorker(env), 
        tts_(tts), 
        text_(std::move(text)), 
        spk_emb_(std::move(spk_emb)), 
        global_emb_(std::move(global_emb)), 
        speed_(speed), 
        lang_(std::move(lang)),
        audio_(nullptr),
        deferred_(Napi::Promise::Deferred::New(env)) {}

  ~OfflineTtsGenerateWithMiocodecLlamaEmbeddingsWorker() {
    if (audio_) {
      SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_);
    }
  }

  void Execute() override {
    SherpaOnnxOfflineTtsMiocodecLlamaEmbeddings embeddings;
    embeddings.speaker_embedding = spk_emb_.data();
    embeddings.speaker_embedding_dim = static_cast<int32_t>(spk_emb_.size());
    embeddings.global_embedding = global_emb_.data();
    embeddings.global_embedding_dim = static_cast<int32_t>(global_emb_.size());

    // We pass nullptr for callback as supporting JS callback in async generation is complex 
    // and handled by TtsGenerateWorker usually, but TtsGenerateWorker handles generic Generate.
    // Here we implement specific one. If progress callback is strictly needed, 
    // it requires TSFN logic similar to TtsGenerateWorker.
    // For now, let's assume no progress callback for this specific async wrapper 
    // unless requested. Given complexity, we'll omit callback support for this async wrapper
    // or reuse TtsGenerateWorker structure if possible? No, signatures differ.
    // So we just run it without callback.

    audio_ = SherpaOnnxOfflineTtsGenerateWithMiocodecLlamaEmbeddings(
        tts_, text_.c_str(), &embeddings, speed_, lang_.c_str(), nullptr, nullptr);
  }

  void OnOK() override {
    Napi::HandleScope scope(Env());
    if (audio_) {
        Napi::Object result = Napi::Object::New(Env());
        
        Napi::ArrayBuffer arrayBuffer =
            Napi::ArrayBuffer::New(Env(), sizeof(float) * audio_->n);
        Napi::Float32Array float32Array =
            Napi::Float32Array::New(Env(), audio_->n, arrayBuffer, 0);
        std::copy(audio_->samples, audio_->samples + audio_->n,
                  float32Array.Data());

        result.Set("samples", float32Array);
        result.Set("sampleRate", audio_->sample_rate);
        deferred_.Resolve(result);
    } else {
       deferred_.Resolve(Env().Undefined());
    }
  }
  
  void OnError(const Napi::Error& e) override {
    deferred_.Reject(Napi::String::New(Env(), e.Message()));
  }

  Napi::Promise Promise() { return deferred_.Promise(); }

 private:
  const SherpaOnnxOfflineTts *tts_;
  std::string text_;
  std::vector<float> spk_emb_;
  std::vector<float> global_emb_;
  float speed_;
  std::string lang_;
  const SherpaOnnxGeneratedAudio *audio_;
  Napi::Promise::Deferred deferred_;
};

class OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsWorker : public Napi::AsyncWorker {
 public:
  OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsWorker(
      Napi::Env env, const SherpaOnnxOfflineTts *tts, std::string source_audio,
      std::vector<float> global_emb, float speed)
      : Napi::AsyncWorker(env),
        tts_(tts),
        source_audio_(std::move(source_audio)),
        global_emb_(std::move(global_emb)),
        speed_(speed),
        audio_(nullptr),
        deferred_(Napi::Promise::Deferred::New(env)) {}

  ~OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsWorker() override {
    if (audio_) {
      SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_);
    }
  }

  void Execute() override {
    SherpaOnnxOfflineTtsMiocodecLlamaEmbeddings embeddings;
    embeddings.speaker_embedding = nullptr;
    embeddings.speaker_embedding_dim = 0;
    embeddings.global_embedding = global_emb_.data();
    embeddings.global_embedding_dim = global_emb_.size();

    audio_ = SherpaOnnxOfflineTtsMiocodecLlamaConvertVoiceWithEmbeddings(
        tts_, source_audio_.c_str(), &embeddings, speed_, nullptr, nullptr);
  }

  void OnOK() override {
    if (!audio_) {
      deferred_.Resolve(Env().Undefined());
    } else {
      Napi::Object obj = Napi::Object::New(Env());
      
      Napi::ArrayBuffer arrayBuffer =
          Napi::ArrayBuffer::New(Env(), sizeof(float) * audio_->n);
      Napi::Float32Array float32Array =
          Napi::Float32Array::New(Env(), audio_->n, arrayBuffer, 0);
      std::copy(audio_->samples, audio_->samples + audio_->n,
                float32Array.Data());

      obj.Set("samples", float32Array);
      obj.Set("sampleRate", audio_->sample_rate);
      deferred_.Resolve(obj);
    }
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(Napi::String::New(Env(), e.Message()));
  }

  Napi::Promise Promise() { return deferred_.Promise(); }

 private:
  const SherpaOnnxOfflineTts *tts_;
  std::string source_audio_;
  std::vector<float> global_emb_;
  float speed_;
  const SherpaOnnxGeneratedAudio *audio_;
  Napi::Promise::Deferred deferred_;
};

static Napi::Value OfflineTtsGenerateWithMiocodecLlamaEmbeddingsAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expect at least 2 arguments: tts, obj").ThrowAsJavaScriptException();
    return env.Null();
  }

  SherpaOnnxOfflineTts *tts =
      reinterpret_cast<SherpaOnnxOfflineTts *>(info[0].As<Napi::External<void>>().Data());
  
  Napi::Object obj = info[1].As<Napi::Object>();
  
  std::string text = obj.Get("text").As<Napi::String>().Utf8Value();
  
  Napi::Float32Array spk = obj.Get("speakerEmbedding").As<Napi::Float32Array>();
  std::vector<float> spk_vec(spk.Data(), spk.Data() + spk.ElementLength());
  
  Napi::Float32Array glob = obj.Get("globalEmbedding").As<Napi::Float32Array>();
  std::vector<float> glob_vec(glob.Data(), glob.Data() + glob.ElementLength());
  
  float speed = 1.0f;
  if (obj.Has("speed")) speed = obj.Get("speed").As<Napi::Number>().FloatValue();
  
  std::string lang = "en-us";
  if (obj.Has("lang")) lang = obj.Get("lang").As<Napi::String>().Utf8Value();

  auto *worker = new OfflineTtsGenerateWithMiocodecLlamaEmbeddingsWorker(
      env, tts, text, std::move(spk_vec), std::move(glob_vec), speed, lang);
      
  worker->Queue();
  return worker->Promise();
}

static Napi::Value OfflineTtsGenerateWithMiocodecLlamaEmbeddingsWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  // similar logic but synchronous
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expect at least 2 arguments: tts, obj").ThrowAsJavaScriptException();
    return env.Null();
  }

  SherpaOnnxOfflineTts *tts =
      reinterpret_cast<SherpaOnnxOfflineTts *>(info[0].As<Napi::External<void>>().Data());
  
  Napi::Object obj = info[1].As<Napi::Object>();
  std::string text = obj.Get("text").As<Napi::String>().Utf8Value();
  
  Napi::Float32Array spk = obj.Get("speakerEmbedding").As<Napi::Float32Array>();
  Napi::Float32Array glob = obj.Get("globalEmbedding").As<Napi::Float32Array>();
  
  SherpaOnnxOfflineTtsMiocodecLlamaEmbeddings embeddings;
  embeddings.speaker_embedding = spk.Data();
  embeddings.speaker_embedding_dim = spk.ElementLength();
  embeddings.global_embedding = glob.Data();
  embeddings.global_embedding_dim = glob.ElementLength();
  
  float speed = 1.0f;
  if (obj.Has("speed")) speed = obj.Get("speed").As<Napi::Number>().FloatValue();
  
  std::string lang = "en-us";
  if (obj.Has("lang")) lang = obj.Get("lang").As<Napi::String>().Utf8Value();

  const auto *audio = SherpaOnnxOfflineTtsGenerateWithMiocodecLlamaEmbeddings(
        tts, text.c_str(), &embeddings, speed, lang.c_str(), nullptr, nullptr);

  if (!audio) return env.Undefined();

  Napi::Object result = Napi::Object::New(env);
  Napi::Float32Array samples = Napi::Float32Array::New(env, audio->n);
  memcpy(samples.Data(), audio->samples, audio->n * sizeof(float));
  result.Set("samples", samples);
  result.Set("sampleRate", audio->sample_rate);
  
  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  return result;
}

static Napi::Value OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsAsyncWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expect at least 2 arguments: tts, obj").ThrowAsJavaScriptException();
    return env.Null();
  }

  SherpaOnnxOfflineTts *tts =
      reinterpret_cast<SherpaOnnxOfflineTts *>(info[0].As<Napi::External<void>>().Data());

  Napi::Object obj = info[1].As<Napi::Object>();
  
  std::string source_audio = "";
  if (obj.Has("sourceAudio")) {
    source_audio = obj.Get("sourceAudio").As<Napi::String>().Utf8Value();
  }

  Napi::Float32Array glob = obj.Get("globalEmbedding").As<Napi::Float32Array>();
  std::vector<float> glob_vec(glob.Data(), glob.Data() + glob.ElementLength());

  float speed = 1.0f;
  if (obj.Has("speed")) speed = obj.Get("speed").As<Napi::Number>().FloatValue();

  auto *worker = new OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsWorker(
      env, tts, source_audio, std::move(glob_vec), speed);

  worker->Queue();
  return worker->Promise();
}

static Napi::Value OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsWrapper(
    const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expect at least 2 arguments: tts, obj").ThrowAsJavaScriptException();
    return env.Null();
  }

  SherpaOnnxOfflineTts *tts =
      reinterpret_cast<SherpaOnnxOfflineTts *>(info[0].As<Napi::External<void>>().Data());

  Napi::Object obj = info[1].As<Napi::Object>();
  
  std::string source_audio = "";
  if (obj.Has("sourceAudio")) {
    source_audio = obj.Get("sourceAudio").As<Napi::String>().Utf8Value();
  }

  Napi::Float32Array glob = obj.Get("globalEmbedding").As<Napi::Float32Array>();

  SherpaOnnxOfflineTtsMiocodecLlamaEmbeddings embeddings;
  embeddings.speaker_embedding = nullptr;
  embeddings.speaker_embedding_dim = 0;
  embeddings.global_embedding = glob.Data();
  embeddings.global_embedding_dim = glob.ElementLength();

  float speed = 1.0f;
  if (obj.Has("speed")) speed = obj.Get("speed").As<Napi::Number>().FloatValue();

  const auto *audio = SherpaOnnxOfflineTtsMiocodecLlamaConvertVoiceWithEmbeddings(
        tts, source_audio.c_str(), &embeddings, speed, nullptr, nullptr);

  if (!audio) return env.Undefined();

  Napi::Object result = Napi::Object::New(env);
  Napi::Float32Array samples = Napi::Float32Array::New(env, audio->n);
  memcpy(samples.Data(), audio->samples, audio->n * sizeof(float));
  result.Set("samples", samples);
  result.Set("sampleRate", audio->sample_rate);

  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  return result;
}
void InitNonStreamingTts(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "createOfflineTts"),
              Napi::Function::New(env, CreateOfflineTtsWrapper));

  exports.Set(Napi::String::New(env, "createOfflineTtsAsync"),
              Napi::Function::New(env, CreateOfflineTtsAsyncWrapper));

  exports.Set(Napi::String::New(env, "getOfflineTtsSampleRate"),
              Napi::Function::New(env, OfflineTtsSampleRateWrapper));

  exports.Set(Napi::String::New(env, "getOfflineTtsNumSpeakers"),
              Napi::Function::New(env, OfflineTtsNumSpeakersWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsGenerate"),
              Napi::Function::New(env, OfflineTtsGenerateWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsGenerateAsync"),
              Napi::Function::New(env, OfflineTtsGenerateAsyncWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsExtractMiocodecEmbeddings"),
              Napi::Function::New(env, OfflineTtsExtractMiocodecLlamaEmbeddingsWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsExtractMiocodecEmbeddingsAsync"),
              Napi::Function::New(env, OfflineTtsExtractMiocodecLlamaEmbeddingsAsyncWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsGenerateWithMiocodecEmbeddings"),
              Napi::Function::New(env, OfflineTtsGenerateWithMiocodecLlamaEmbeddingsWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsGenerateWithMiocodecEmbeddingsAsync"),
              Napi::Function::New(env, OfflineTtsGenerateWithMiocodecLlamaEmbeddingsAsyncWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsConvertVoiceWithMiocodecEmbeddings"),
              Napi::Function::New(env, OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsWrapper));

  exports.Set(Napi::String::New(env, "offlineTtsConvertVoiceWithMiocodecEmbeddingsAsync"),
              Napi::Function::New(env, OfflineTtsConvertVoiceWithMiocodecLlamaEmbeddingsAsyncWrapper));
}
