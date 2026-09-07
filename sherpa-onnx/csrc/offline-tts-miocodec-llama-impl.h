// sherpa-onnx/csrc/offline-tts-miocodec-llama-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/kokoro-multi-lang-lexicon.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-miocodec-llama-model.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "kaldifst/csrc/text-normalizer.h"
#include "fst/extensions/far/far.h"
#include "kaldifst/csrc/kaldi-fst-io.h"

namespace sherpa_onnx {

class OfflineTtsMiocodecLlamaImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsMiocodecLlamaImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMiocodecLlamaModel>(config.model)) {
    InitLexicon();
    InitFst(config);
  }

  template <typename Manager>
  OfflineTtsMiocodecLlamaImpl(Manager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsMiocodecLlamaModel>(mgr,
                                                               config.model)) {
    InitLexicon(mgr);
    InitFst(config);
  }

  int32_t SampleRate() const override { return model_->SampleRate(); }

  int32_t NumSpeakers() const override {
    // MioCodec-LLaMA is a zero-shot TTS model
    return -1;
  }

  // Zero-shot TTS with reference audio for speaker embedding.
  // audio_dir  — path to a reference WAV file for speaker cloning
  // lang       — language code, e.g. "en", "zh", "ja", "ko", "fr", "de", "es"
  // exaggeration — unused (kept for interface compatibility)
  const OfflineTtsConfig &GetConfig() const { return config_; }

  struct MiocodecLlamaEmbeddings {
    std::vector<float> speaker_embedding;  // 192-dim
    std::vector<float> global_embedding;   // 512-dim
  };

  // Extract both embeddings from a reference audio file
  MiocodecLlamaEmbeddings ExtractEmbeddings(const std::string &audio_dir) const {
    if (audio_dir.empty()) {
      SHERPA_ONNX_LOGE("MioCodec-LLaMA: audio_dir is empty.");
      return {};
    }

    bool is_ok = false;
    int32_t ref_sr = -1;
    std::vector<float> ref_samples = ReadWave(audio_dir, &ref_sr, &is_ok);
    if (!is_ok) {
      SHERPA_ONNX_LOGE("Failed to read reference audio: %s", audio_dir.c_str());
      return {};
    }

    // Cam++ (16k) and MioCodec (model's sample rate)
    std::vector<float> ref_16k = Resample(ref_samples, ref_sr, 16000);
    std::vector<float> ref_miocodec = Resample(ref_samples, ref_sr, SampleRate());

    MiocodecLlamaEmbeddings ans;
    ans.speaker_embedding = model_->ExtractSpeakerEmbedding(
        ref_16k.data(), static_cast<int32_t>(ref_16k.size()));

    auto mio_feats = model_->ExtractMiocodecFeatures(
        ref_miocodec.data(), static_cast<int32_t>(ref_miocodec.size()));
    ans.global_embedding = std::move(mio_feats.global_embedding);

    return ans;
  }

  GeneratedAudio Generate(const std::string &text, const std::string &audio_dir,
                          float speed, const std::string &lang,
                          float /*exaggeration*/,
                          GeneratedAudioCallback callback) const override {
    auto embeddings = ExtractEmbeddings(audio_dir);
    if (embeddings.speaker_embedding.empty() ||
        embeddings.global_embedding.empty()) {
      return {};
    }

    return GenerateWithEmbeddings(text, embeddings.speaker_embedding,
                                  embeddings.global_embedding, speed, lang,
                                  callback);
  }

  GeneratedAudio GenerateWithEmbeddings(
      const std::string &text, const std::vector<float> &speaker_embedding,
      const std::vector<float> &global_embedding, float speed,
      const std::string &lang, GeneratedAudioCallback callback) const {
    if (!lexicon_) {
      SHERPA_ONNX_LOGE("MioCodec-LLaMA: G2P lexicon not initialised.");
      return {};
    }

    int32_t lang_id = GetLanguageId(lang);
    std::string kokoro_voice = LangToKokoroVoice(lang);

    std::string normalized_text = text;
    // Apply FST text normalization first (for number/date conversion)
    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        normalized_text = tn->Normalize(normalized_text);
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("After FST normalizing: %s", normalized_text.c_str());
        }
      }
    }

    auto token_ids_vec = lexicon_->ConvertTextToTokenIds(normalized_text, kokoro_voice);
    if (token_ids_vec.empty()) {
      SHERPA_ONNX_LOGE("G2P produced no tokens for input text: %s", normalized_text.c_str());
      return {};
    }

    const auto &cfg = config_.model.miocodec_llama;
    std::vector<int64_t> all_phonemes;

    // Concatenate all Kokoro chunks into a single prompt for GPT
    for (const auto &chunk : token_ids_vec) {
      if (chunk.tokens.empty()) continue;
      for (auto t : chunk.tokens) {
        if (t != 0) {  // Remove BOS/EOS/Space token 0
          all_phonemes.push_back(t);
        }
      }
    }

    if (config_.model.debug) {
      std::ostringstream os;
      os << "all_phonemes (size " << all_phonemes.size() << "): [";
      for (size_t i = 0; i < all_phonemes.size(); ++i) {
        os << all_phonemes[i] << (i + 1 == all_phonemes.size() ? "" : ", ");
      }
      os << "]";
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    if (all_phonemes.empty()) {
      return {};
    }

    std::vector<int64_t> audio_tokens = model_->GenerateAudioTokens(
        all_phonemes, speaker_embedding, lang_id, cfg.temperature, cfg.top_p,
        cfg.max_tokens, cfg.repetition_penalty);

    if (audio_tokens.empty()) {
      return {};
    }

    std::vector<float> all_samples = model_->DecodeMiocodec(
        audio_tokens.data(), static_cast<int32_t>(audio_tokens.size()),
        global_embedding.data(), static_cast<int32_t>(global_embedding.size()));

    if (std::abs(speed - 1.0f) > 0.01f) {
      int32_t target_sr = static_cast<int32_t>(SampleRate() / speed);
      all_samples = Resample(all_samples, SampleRate(), target_sr);
    }

    if (callback) {
      float progress = 1.0f;
      callback(all_samples.data(), static_cast<int32_t>(all_samples.size()), progress);
    }

    GeneratedAudio ans;
    ans.sample_rate = SampleRate();
    ans.samples = std::move(all_samples);
    return ans;
  }

  GeneratedAudio ConvertVoiceWithEmbeddings(
      const std::string &source_audio_dir,
      const std::vector<float> &global_embedding,
      float speed,
      GeneratedAudioCallback callback) const {
    if (global_embedding.empty()) {
      SHERPA_ONNX_LOGE("MioCodec-LLaMA VC: global_embedding is empty.");
      return {};
    }

    if (source_audio_dir.empty()) {
      SHERPA_ONNX_LOGE("MioCodec-LLaMA VC: source_audio_dir is empty.");
      return {};
    }

    int32_t src_sr = -1;
    bool is_ok = false;
    std::vector<float> src_samples = ReadWave(source_audio_dir, &src_sr, &is_ok);
    if (!is_ok) {
      SHERPA_ONNX_LOGE("Failed to read source audio for VC: %s", source_audio_dir.c_str());
      return {};
    }

    std::vector<float> src_miocodec = Resample(src_samples, src_sr, SampleRate());
    auto mio_feats = model_->ExtractMiocodecFeatures(
        src_miocodec.data(), static_cast<int32_t>(src_miocodec.size()));

    if (mio_feats.content_indices.empty()) {
      SHERPA_ONNX_LOGE("MioCodec-LLaMA VC: extracted no content indices from source.");
      return {};
    }

    std::vector<float> all_samples = model_->DecodeMiocodec(
        mio_feats.content_indices.data(), static_cast<int32_t>(mio_feats.content_indices.size()),
        global_embedding.data(), static_cast<int32_t>(global_embedding.size()));

    if (std::abs(speed - 1.0f) > 0.01f) {
      int32_t target_sr = static_cast<int32_t>(SampleRate() / speed);
      all_samples = Resample(all_samples, SampleRate(), target_sr);
    }

    if (callback) {
      float progress = 1.0f;
      callback(all_samples.data(), static_cast<int32_t>(all_samples.size()), progress);
    }

    GeneratedAudio ans;
    ans.sample_rate = SampleRate();
    ans.samples = std::move(all_samples);
    return ans;
  }

 private:
  void InitLexicon() {
    const auto &cfg = config_.model.miocodec_llama;
    if (cfg.tokens.empty()) return;

    OfflineTtsKokoroModelMetaData meta;
    meta.voice = "en-us";
    meta.max_token_len = 512;
    meta.has_espeak = !cfg.g2p_model.empty();

    lexicon_ = std::make_unique<KokoroMultiLangLexicon>(
        cfg.g2p_model, cfg.tokens, cfg.lexicon, meta,
        config_.model.debug);
  }

  template <typename Manager>
  void InitLexicon(Manager *mgr) {
    const auto &cfg = config_.model.miocodec_llama;
    if (cfg.tokens.empty()) return;

    OfflineTtsKokoroModelMetaData meta;
    meta.voice = "en-us";
    meta.max_token_len = 512;
    meta.has_espeak = !cfg.g2p_model.empty();

    lexicon_ = std::make_unique<KokoroMultiLangLexicon>(
        mgr, cfg.g2p_model, cfg.tokens, cfg.lexicon, meta,
        config_.model.debug);
  }

  // Resample audio from src_sr to dst_sr
  static std::vector<float> Resample(const std::vector<float> &samples,
                                     int32_t src_sr, int32_t dst_sr) {
    if (src_sr == dst_sr) return samples;
    float cutoff_hz = std::min(src_sr, dst_sr) / 2.0f * 0.95f;
    LinearResample resampler(src_sr, dst_sr, cutoff_hz, /*num_zeros=*/8);
    std::vector<float> out;
    resampler.Resample(samples.data(), samples.size(), /*flush=*/true, &out);
    return out;
  }

  void InitFst(const OfflineTtsConfig &config) {
    // Load FST text normalizers (for number/date conversion)
    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("miocodec-llama rule fst: %s", f.c_str());
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }

    if (!config.rule_fars.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fars, ",", false, &files);
      tn_list_.reserve(files.size() + tn_list_.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("miocodec-llama rule far: %s", f.c_str());
        }
        std::unique_ptr<fst::FarReader<fst::StdArc>> reader(
            fst::FarReader<fst::StdArc>::Open(f));
        for (; !reader->Done(); reader->Next()) {
          std::unique_ptr<fst::StdConstFst> r(
              fst::CastOrConvertToConstFst(reader->GetFst()->Copy()));
          tn_list_.push_back(
              std::make_unique<kaldifst::TextNormalizer>(std::move(r)));
        }
      }
    }
  }

  // Map language string to the integer language ID expected by the GPT model
  static int32_t GetLanguageId(const std::string &lang) {
    static const std::unordered_map<std::string, int32_t> kMap = {
        {"en", 13011},  {"en-us", 13011}, {"en-gb", 13011}, 
        {"cmn", 13010}, {"zh", 13010},    {"zh-cn", 13010}, 
        {"ja", 13012},
    };
    auto it = kMap.find(lang);
    return (it != kMap.end()) ? it->second : 13010; // default to zh
  }

  // Map sherpa language code to the Kokoro voice name for G2P lookup
  static std::string LangToKokoroVoice(const std::string &lang) {
    if (lang == "zh" || lang == "cmn" || lang == "zh-cn") return "cmn";
    if (lang == "ja") return "ja";
    if (lang == "ko") return "ko";
    if (lang == "fr") return "fr";
    if (lang == "de") return "de";
    if (lang == "es") return "es";
    if (lang == "pt" || lang == "pt-br") return "pt-br";
    return "en-us";
  }

  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsMiocodecLlamaModel> model_;
  std::unique_ptr<KokoroMultiLangLexicon> lexicon_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_IMPL_H_
