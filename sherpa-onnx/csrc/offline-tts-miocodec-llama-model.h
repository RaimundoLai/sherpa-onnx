// sherpa-onnx/csrc/offline-tts-miocodec-llama-model.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_MODEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"

namespace sherpa_onnx {

// Features extracted by ExtractMiocodecFeatures().
// Holds content indices (used as audio token source for VC) and the
// global speaker embedding (used to condition the decoder).
struct MiocodecFeatures {
  // (T_tokens,) — content indices from MioCodec encoder
  std::vector<int64_t> content_indices;

  // Flat global embedding vector from MioCodec encoder
  std::vector<float> global_embedding;

  // Dimensionality of global_embedding (e.g. 512)
  int32_t global_embedding_dim = 0;
};

// OfflineTtsMiocodecLlamaModel loads and runs the four ONNX models that make
// up the MioCodec-LLaMA TTS pipeline.
//
// The key design decision is that the two embedding-extraction operations
// (CAM++ speaker embedding and MioCodec global embedding) are exposed as
// *separate*, pre-computable methods so callers can cache results and avoid
// re-running them for every utterance.
class OfflineTtsMiocodecLlamaModel {
 public:
  explicit OfflineTtsMiocodecLlamaModel(const OfflineTtsModelConfig &config);

  template <typename Manager>
  OfflineTtsMiocodecLlamaModel(Manager *mgr,
                               const OfflineTtsModelConfig &config);

  ~OfflineTtsMiocodecLlamaModel();

  // ---------------------------------------------------------------------------
  // Standalone embedding extraction  (pre-computable / cacheable)
  // ---------------------------------------------------------------------------

  // Run CAM++ on mono 16 kHz audio to extract a 192-dim speaker embedding.
  //
  // The model internally computes 80-dim log Mel fbank features from the raw
  // waveform before running CAM++.
  //
  // @param audio_samples  Raw 16 kHz mono PCM samples (float, [-1, 1])
  // @param audio_len      Number of samples
  // @return               Float vector of length 192
  std::vector<float> ExtractSpeakerEmbedding(const float *audio_samples,
                                             int32_t audio_len);

  // Run MioCodec encoder on mono 24 kHz audio to obtain content indices and
  // a global embedding.  Both outputs can be cached and reused.
  //
  // @param audio_samples  Raw 24 kHz mono PCM samples (float, [-1, 1])
  // @param audio_len      Number of samples
  // @return               MiocodecFeatures with content_indices and
  //                       global_embedding
  MiocodecFeatures ExtractMiocodecFeatures(const float *audio_samples,
                                           int32_t audio_len);

  // ---------------------------------------------------------------------------
  // Core generation
  // ---------------------------------------------------------------------------

  // Run the GPT-LinaCodec language model autoregressively (with KV cache) to
  // produce a sequence of audio tokens.
  //
  // Inputs are the phoneme token IDs (from G2P), a pre-computed 192-dim speaker
  // embedding (from ExtractSpeakerEmbedding), and a language ID.
  //
  // @param phoneme_ids        Sequence of phoneme token IDs
  // @param speaker_embedding  Pre-computed 192-dim CAM++ embedding
  // @param language_id        Language token ID to prepend (0 = English)
  // @param temperature        Softmax temperature for sampling
  // @param top_p              Nucleus sampling threshold
  // @param max_tokens         Maximum number of audio tokens to generate
  // @param repetition_penalty Repetition penalty (1.0 = no penalty)
  // @return                   Flat list of audio token IDs
  std::vector<int64_t> GenerateAudioTokens(
      const std::vector<int64_t> &phoneme_ids,
      const std::vector<float> &speaker_embedding, int32_t language_id,
      float temperature, float top_p, int32_t max_tokens,
      float repetition_penalty);

  // Decode a sequence of content indices together with a global embedding back
  // into a waveform using the MioCodec decoder (24 kHz output).
  //
  // @param content_indices  Audio/content token IDs (T,)
  // @param num_tokens       Length of content_indices
  // @param global_embedding Pre-computed global embedding from MioCodec encoder
  // @param global_dim       Dimensionality of global_embedding
  // @return                 24 kHz mono PCM samples
  std::vector<float> DecodeMiocodec(const int64_t *content_indices,
                                    int32_t num_tokens,
                                    const float *global_embedding,
                                    int32_t global_dim);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MIOCODEC_LLAMA_MODEL_H_
