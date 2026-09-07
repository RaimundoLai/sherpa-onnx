// sherpa-onnx/csrc/offline-tts-miocodec-llama-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-miocodec-llama-model.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sherpa_onnx {

// ---------------------------------------------------------------------------
// Hardcoded model architecture constants for the MioCodec-LLaMA model.
static constexpr int32_t kEosId = 13001;           // EOS token for audio generation
static constexpr int32_t kSepId = 13002;           // SEP token between phonemes and audio
static constexpr int32_t kAudioOffset = 200;       // Offset: audio_token = vocab_id - kAudioOffset
static constexpr int32_t kAudioVocabSize = 12800;  // MioCodec codebook size
static constexpr int32_t kNumLayers = 12;      // Number of transformer layers
static constexpr int32_t kNumHeads = 12;       // Number of attention heads
static constexpr int32_t kHiddenDim = 768;     // Model hidden dimension
static constexpr int32_t kHeadDim = kHiddenDim / kNumHeads;  // = 64

// Language token IDs (prepended to phoneme sequence)
static const std::unordered_map<std::string, int32_t> kLangIdMap = {
    {"en", 0}, {"en-us", 0}, {"cmn", 1}, {"zh", 1}, {"ja", 2},
    {"ko", 3}, {"fr", 4},    {"de", 5},  {"es", 6}, {"pt", 7},
};

// ---------------------------------------------------------------------------
// Minimal NPZ / NPY reader
// ---------------------------------------------------------------------------
// NPZ files are ZIP archives of NPY files. We implement a minimal parser that:
// 1. Reads the ZIP local file headers to find each array
// 2. Parses the NPY header to get dtype, shape
// 3. Copies the raw float data into a std::vector<float>

namespace {

// Read a little-endian uint16 / uint32 from a byte buffer
inline uint16_t ReadU16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
inline uint32_t ReadU32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

struct NpzArray {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

// Parse NPY v1.0 / v2.0 magic + header, extract float32 data.
// Returns false if the dtype is not float32 or parsing fails.
static bool ParseNpy(const uint8_t *buf, size_t buf_len, NpzArray *out) {
  // Magic: \x93NUMPY
  if (buf_len < 10) return false;
  if (buf[0] != 0x93 || buf[1] != 'N' || buf[2] != 'U' || buf[3] != 'M' ||
      buf[4] != 'P' || buf[5] != 'Y') {
    SHERPA_ONNX_LOGE("Not a valid .npy file (bad magic)");
    return false;
  }
  // uint8 major, minor
  uint8_t major = buf[6];
  size_t header_len_field_size = (major == 1) ? 2 : 4;
  size_t hdr_offset = 8;
  if (buf_len < hdr_offset + header_len_field_size) return false;

  uint32_t header_len = 0;
  if (major == 1) {
    header_len = ReadU16(buf + hdr_offset);
  } else {
    header_len = ReadU32(buf + hdr_offset);
  }
  hdr_offset += header_len_field_size;

  if (buf_len < hdr_offset + header_len) {
    SHERPA_ONNX_LOGE("NPY file truncated in header");
    return false;
  }

  std::string header(reinterpret_cast<const char *>(buf + hdr_offset),
                     header_len);
  size_t data_offset = hdr_offset + header_len;

  // Only support float32
  if (header.find("'<f4'") == std::string::npos &&
      header.find("\"<f4\"") == std::string::npos) {
    SHERPA_ONNX_LOGE(
        "embeddings.npz: Only float32 ('<f4') arrays are supported");
    return false;
  }

  // Parse shape from header string, e.g. 'shape': (512, 192),
  auto shape_pos = header.find("'shape'");
  if (shape_pos == std::string::npos) shape_pos = header.find("\"shape\"");
  if (shape_pos == std::string::npos) {
    SHERPA_ONNX_LOGE("Cannot find 'shape' in NPY header: %s", header.c_str());
    return false;
  }
  auto lparen = header.find('(', shape_pos);
  auto rparen = header.find(')', shape_pos);
  if (lparen == std::string::npos || rparen == std::string::npos) return false;

  std::string shape_str = header.substr(lparen + 1, rparen - lparen - 1);
  out->shape.clear();
  std::stringstream ss(shape_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    // strip whitespace
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t,") + 1);
    if (!token.empty()) {
      out->shape.push_back(std::stoll(token));
    }
  }

  int64_t num_elems = 1;
  for (auto d : out->shape) num_elems *= d;

  size_t expected_bytes = static_cast<size_t>(num_elems) * sizeof(float);
  if (buf_len < data_offset + expected_bytes) {
    SHERPA_ONNX_LOGE("NPY data truncated: need %zu bytes, have %zu",
                     expected_bytes, buf_len - data_offset);
    return false;
  }

  const float *fdata = reinterpret_cast<const float *>(buf + data_offset);
  out->data.assign(fdata, fdata + num_elems);
  return true;
}

// Parse NPZ (ZIP) to extract named float32 arrays.
// Fills `arrays` map: stripped_name (without .npy) -> NpzArray.
static bool ParseNpz(
    const uint8_t *buf, size_t buf_len,
    std::unordered_map<std::string, NpzArray> *arrays) {
  // ZIP local file header signature = 0x04034b50
  // We scan through all local file headers.
  size_t pos = 0;
  while (pos + 30 <= buf_len) {
    uint32_t sig = ReadU32(buf + pos);
    if (sig != 0x04034b50u) {
      ++pos;
      continue;
    }
    // Local file header fields:
    // offset 4: version needed (2), flags (2), compression (2)
    uint16_t compression = ReadU16(buf + pos + 8);
    uint32_t compressed_size = ReadU32(buf + pos + 18);
    uint32_t uncompressed_size = ReadU32(buf + pos + 22);
    uint16_t fname_len = ReadU16(buf + pos + 26);
    uint16_t extra_len = ReadU16(buf + pos + 28);

    if (compression != 0) {
      // Compressed entries — skip (NPZ saved by numpy default uses DEFLATE
      // but we expect uncompressed for simplicity; numpy uses ZIP_STORED when
      // allow_pickle=False and the array fits in memory)
      pos += 30 + fname_len + extra_len + compressed_size;
      continue;
    }

    size_t data_start = pos + 30 + fname_len + extra_len;
    if (data_start + uncompressed_size > buf_len) break;

    std::string fname(reinterpret_cast<const char *>(buf + pos + 30), fname_len);
    // Strip ".npy" suffix
    std::string key = fname;
    if (key.size() >= 4 && key.substr(key.size() - 4) == ".npy") {
      key = key.substr(0, key.size() - 4);
    }

    NpzArray arr;
    if (ParseNpy(buf + data_start, uncompressed_size, &arr)) {
      (*arrays)[key] = std::move(arr);
    }

    pos = data_start + uncompressed_size;
  }
  return !arrays->empty();
}

// Softmax over a float array in-place
static void Softmax(float *data, int32_t n) {
  float max_val = *std::max_element(data, data + n);
  float sum = 0;
  for (int32_t i = 0; i < n; ++i) {
    data[i] = std::exp(data[i] - max_val);
    sum += data[i];
  }
  for (int32_t i = 0; i < n; ++i) data[i] /= sum;
}

// Top-p (nucleus) sampling: returns a sampled index given logits.
static int32_t SampleTopP(const float *logits, int32_t vocab_size,
                           float temperature, float top_p,
                           std::mt19937 *rng) {
  std::vector<float> probs(logits, logits + vocab_size);
  // Apply temperature
  if (temperature > 1e-6f) {
    for (auto &v : probs) v /= temperature;
  }
  Softmax(probs.data(), vocab_size);

  // Sort indices by descending prob
  std::vector<int32_t> idx(vocab_size);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });

  // Keep top tokens accumulating to top_p
  float cumsum = 0;
  std::vector<float> top_probs;
  std::vector<int32_t> top_idx;
  for (int32_t i : idx) {
    top_probs.push_back(probs[i]);
    top_idx.push_back(i);
    cumsum += probs[i];
    if (cumsum >= top_p) break;
  }

  // Re-normalize
  float total = std::accumulate(top_probs.begin(), top_probs.end(), 0.0f);
  for (auto &v : top_probs) v /= total;

  std::discrete_distribution<int32_t> dist(top_probs.begin(), top_probs.end());
  return top_idx[dist(*rng)];
}

// Matrix-vector product: y = W * x + b
// W shape: (out_dim, in_dim), x shape: (in_dim,), b shape: (out_dim,)
static std::vector<float> MatVecAdd(const float *W, const float *x,
                                    const float *b, int32_t out_dim,
                                    int32_t in_dim) {
  std::vector<float> y(out_dim, 0.0f);
  for (int32_t i = 0; i < out_dim; ++i) {
    float sum = b ? b[i] : 0.0f;
    for (int32_t j = 0; j < in_dim; ++j) {
      sum += W[i * in_dim + j] * x[j];
    }
    y[i] = sum;
  }
  return y;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl class
// ---------------------------------------------------------------------------
class OfflineTtsMiocodecLlamaModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        rng_(42) {
    // Load model bytes from filesystem
    auto gpt_buf = ReadFile(config.miocodec_llama.model);
    auto campplus_buf = ReadFile(config.miocodec_llama.campplus_model);
    auto enc_buf = ReadFile(config.miocodec_llama.miocodec_encoder);
    auto dec_buf = ReadFile(config.miocodec_llama.miocodec_decoder);
    auto emb_buf = ReadFile(config.miocodec_llama.embeddings);

    Init(gpt_buf.data(), gpt_buf.size(), campplus_buf.data(),
         campplus_buf.size(), enc_buf.data(), enc_buf.size(), dec_buf.data(),
         dec_buf.size(), emb_buf.data(), emb_buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        rng_(42) {
    auto gpt_buf = ReadFile(mgr, config.miocodec_llama.model);
    auto campplus_buf = ReadFile(mgr, config.miocodec_llama.campplus_model);
    auto enc_buf = ReadFile(mgr, config.miocodec_llama.miocodec_encoder);
    auto dec_buf = ReadFile(mgr, config.miocodec_llama.miocodec_decoder);
    auto emb_buf = ReadFile(mgr, config.miocodec_llama.embeddings);

    Init(gpt_buf.data(), gpt_buf.size(), campplus_buf.data(),
         campplus_buf.size(), enc_buf.data(), enc_buf.size(), dec_buf.data(),
         dec_buf.size(), emb_buf.data(), emb_buf.size());
  }

  int32_t SampleRate() const { return sample_rate_; }

  // ------ Embedding extraction -----------------------------------------------

  std::vector<float> ExtractSpeakerEmbedding(const float *audio_samples,
                                             int32_t audio_len) {
    // CAM++ expects 80-dim log-mel fbank features: shape (1, T, 80)
    // We compute fbank internally using a 25ms window / 10ms shift at 16 kHz.
    auto fbank = ComputeFbank(audio_samples, audio_len, /*sample_rate=*/16000);
    int32_t num_frames = static_cast<int32_t>(fbank.size()) / 80;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> fbank_shape = {1, num_frames, 80};
    Ort::Value fbank_tensor = Ort::Value::CreateTensor(
        memory_info, fbank.data(), fbank.size(), fbank_shape.data(),
        fbank_shape.size());

    try {
      auto out = sess_campplus_->Run({}, campplus_input_names_ptr_.data(),
                                     &fbank_tensor, 1,
                                     campplus_output_names_ptr_.data(),
                                     campplus_output_names_ptr_.size());
      // Output: (1, 192)
      const float *data = out[0].GetTensorData<float>();
      auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
      int32_t emb_dim = 1;
      for (auto d : shape) emb_dim *= d;
      return std::vector<float>(data, data + emb_dim);
    } catch (const Ort::Exception& e) {
      SHERPA_ONNX_LOGE("ONNX Runtime Exception in ExtractSpeakerEmbedding: %s", e.what());
      return {};
    }
  }

  MiocodecFeatures ExtractMiocodecFeatures(const float *audio_samples,
                                           int32_t audio_len) {
    // MioCodec encoder expects shape (1, T) at 24 kHz
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> audio_shape = {1, audio_len};
    Ort::Value audio_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(audio_samples),
        static_cast<size_t>(audio_len), audio_shape.data(),
        audio_shape.size());

    try {
      auto out = sess_miocodec_enc_->Run({}, enc_input_names_ptr_.data(),
                                         &audio_tensor, 1,
                                         enc_output_names_ptr_.data(),
                                         enc_output_names_ptr_.size());

      // Expected outputs: content_indices (1, T_tokens) and global_emb (1, C)
      MiocodecFeatures feats;

      // Output 0: content indices (int64 or float — cast to int64)
      {
        auto &t = out[0];
        auto shape = t.GetTensorTypeAndShapeInfo().GetShape();
        int64_t n = 1;
        for (auto d : shape) n *= d;
        auto type = t.GetTensorTypeAndShapeInfo().GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          const int64_t *p = t.GetTensorData<int64_t>();
          feats.content_indices.assign(p, p + n);
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          const int32_t *p = t.GetTensorData<int32_t>();
          feats.content_indices.assign(p, p + n);
        } else {
          // Some exported models use float indices — round to int
          const float *p = t.GetTensorData<float>();
          feats.content_indices.resize(n);
          for (int64_t i = 0; i < n; ++i) {
            feats.content_indices[i] =
                static_cast<int64_t>(std::round(p[i]));
          }
        }
      }

      // Output 1: global embedding (1, C)
      {
        auto &t = out[1];
        auto shape = t.GetTensorTypeAndShapeInfo().GetShape();
        int64_t c = 1;
        for (auto d : shape) c *= d;
        feats.global_embedding_dim = static_cast<int32_t>(c);
        const float *p = t.GetTensorData<float>();
        feats.global_embedding.assign(p, p + c);
      }

      return feats;
    } catch (const Ort::Exception& e) {
      SHERPA_ONNX_LOGE("ONNX Runtime Exception in ExtractMiocodecFeatures: %s", e.what());
      return {};
    }
  }

  // ------ Core generation ----------------------------------------------------

  std::vector<int64_t> GenerateAudioTokens(
      const std::vector<int64_t> &phoneme_ids,
      const std::vector<float> &speaker_embedding, int32_t language_id,
      float temperature, float top_p, int32_t max_tokens,
      float repetition_penalty) {
    // Step 1: Compute speaker token embedding.
    //   speaker_token = spk_emb @ spk_proj_weight.T + spk_proj_bias
    //   spk_emb shape: (192,)
    //   spk_proj_weight shape: (hidden_dim, 192)
    //   Result: (hidden_dim,)
    int32_t spk_dim = static_cast<int32_t>(speaker_embedding.size());
    std::vector<float> spk_token =
        MatVecAdd(spk_proj_weight_.data(), speaker_embedding.data(),
                  spk_proj_bias_.data(), kHiddenDim, spk_dim);

    // Step 2: Build the prefill input_embeds
    //   = [spk_token, lang_emb, phoneme_embs..., sep_emb]
    //   where each token embedding is looked up from tok_emb_weight.

    // Build token sequence: [LANG_ID, phoneme_ids..., SEP_ID]
    std::vector<int64_t> text_tokens;
    text_tokens.push_back(static_cast<int64_t>(language_id));
    text_tokens.insert(text_tokens.end(), phoneme_ids.begin(),
                       phoneme_ids.end());
    text_tokens.push_back(static_cast<int64_t>(kSepId));

    // Construct prefill embeddings: shape (1, 1 + T_text, hidden_dim)
    int32_t T_text = static_cast<int32_t>(text_tokens.size());
    int32_t T_prefill = 1 + T_text;  // spk_token + text_tokens
    std::vector<float> prefill_embeds(T_prefill * kHiddenDim, 0.0f);

    // First token = speaker token
    std::copy(spk_token.begin(), spk_token.end(), prefill_embeds.begin());

    // Remaining tokens: lookup in tok_emb_weight (vocab_size, hidden_dim)
    for (int32_t i = 0; i < T_text; ++i) {
      int64_t tok_id = text_tokens[i];
      const float *emb = tok_emb_weight_.data() +
                         static_cast<size_t>(tok_id) * kHiddenDim;
      std::copy(emb, emb + kHiddenDim,
                prefill_embeds.begin() + (1 + i) * kHiddenDim);
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // Step 3: Prefill — run GPT with all text tokens to obtain KV cache.
    //   Input: input_embeds (1, T_prefill, hidden_dim)
    //          position_ids (1, T_prefill)
    //          empty past_key_values (2 * kNumLayers tensors of shape
    //            (1, kNumHeads, 0, kHeadDim))
    std::array<int64_t, 3> emb_shape = {1, T_prefill, kHiddenDim};
    Ort::Value emb_tensor = Ort::Value::CreateTensor(
        memory_info, prefill_embeds.data(), prefill_embeds.size(),
        emb_shape.data(), emb_shape.size());

    std::vector<int64_t> pos_ids(T_prefill);
    std::iota(pos_ids.begin(), pos_ids.end(), 0);
    std::array<int64_t, 1> pos_shape = {T_prefill};
    Ort::Value pos_tensor = Ort::Value::CreateTensor(
        memory_info, pos_ids.data(), pos_ids.size(), pos_shape.data(),
        pos_shape.size());

    // Create empty KV cache: shape (1, num_heads, 0, head_dim)
    std::array<int64_t, 4> kv_shape = {1, kNumHeads, 0, kHeadDim};
    std::vector<std::vector<float>> kv_storage(2 * kNumLayers);
    std::vector<Ort::Value> kv_inputs;
    for (int32_t i = 0; i < 2 * kNumLayers; ++i) {
      kv_inputs.push_back(Ort::Value::CreateTensor(
          memory_info, kv_storage[i].data(), 0, kv_shape.data(), 4));
    }

    // Assemble all inputs: [input_embeds, position_ids, past_key_values...]
    std::vector<Ort::Value> gpt_inputs;
    gpt_inputs.push_back(std::move(emb_tensor));
    gpt_inputs.push_back(std::move(pos_tensor));
    for (auto &kv : kv_inputs) {
      gpt_inputs.push_back(std::move(kv));
    }

    auto prefill_out =
        sess_gpt_->Run({}, gpt_input_names_ptr_.data(), gpt_inputs.data(),
                       gpt_inputs.size(), gpt_output_names_ptr_.data(),
                       gpt_output_names_ptr_.size());

    // Outputs: [logits (1, T_prefill, vocab), new_kv (2*num_layers tensors)]
    // We only care about the last logit position for sampling.
    std::vector<int64_t> generated_tokens;
    generated_tokens.reserve(max_tokens);

    // Extract KV cache tensors (outputs 1..2*kNumLayers+1)
    // They have shape (1, num_heads, T_prefill, head_dim)
    std::vector<Ort::Value> past_kv;
    past_kv.reserve(2 * kNumLayers);
    for (int32_t i = 1; i <= 2 * kNumLayers; ++i) {
      past_kv.push_back(std::move(prefill_out[i]));
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Prefill done. T_prefill=%d, Max tokens=%d", T_prefill, max_tokens);
    }

    // Get the last logit to sample the first audio token
    {
      const float *logits = prefill_out[0].GetTensorData<float>();
      auto logits_shape =
          prefill_out[0].GetTensorTypeAndShapeInfo().GetShape();
      int32_t V = static_cast<int32_t>(logits_shape[2]);

      // Only care about the last position
      const float *last_logit =
          logits + (T_prefill - 1) * static_cast<int64_t>(V);

      // Apply repetition penalty to already-generated tokens
      std::vector<float> adj_logits(last_logit, last_logit + V);
      ApplyRepetitionPenalty(adj_logits.data(), V, generated_tokens,
                             repetition_penalty);

      int32_t sampled = SampleTopP(adj_logits.data(), V, temperature, top_p,
                                   &rng_);
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Prefill sampled token: %d (EOS=%d, AudioOffset=%d)", sampled, kEosId, kAudioOffset);
      }
      if (sampled == kEosId) {
        // Nothing to decode
        if (config_.debug) SHERPA_ONNX_LOGE("First token is EOS. Stopping generation.");
        return {};
      }
      generated_tokens.push_back(static_cast<int64_t>(sampled));
    }

    // Step 4: Decode loop — generate one audio token at a time
    for (int32_t step = 0; step < max_tokens - 1; ++step) {
      int64_t last_tok = generated_tokens.back();
      if (last_tok == kEosId) {
        generated_tokens.pop_back();  // remove EOS
        if (config_.debug) SHERPA_ONNX_LOGE("Hit EOS at step %d. Ending decode loop.", step);
        break;
      }

      // Look up embedding of last token
      const float *emb_ptr =
          tok_emb_weight_.data() +
          static_cast<size_t>(last_tok) * kHiddenDim;
      std::vector<float> tok_emb(emb_ptr, emb_ptr + kHiddenDim);

      std::array<int64_t, 3> step_emb_shape = {1, 1, kHiddenDim};
      Ort::Value step_emb_tensor = Ort::Value::CreateTensor(
          memory_info, tok_emb.data(), tok_emb.size(), step_emb_shape.data(),
          step_emb_shape.size());

      int64_t step_pos_val = T_prefill + step;
      std::array<int64_t, 1> step_pos_shape = {1};
      Ort::Value step_pos_tensor = Ort::Value::CreateTensor(
          memory_info, &step_pos_val, 1, step_pos_shape.data(),
          step_pos_shape.size());

      // Assemble decode inputs
      std::vector<Ort::Value> step_inputs;
      step_inputs.push_back(std::move(step_emb_tensor));
      step_inputs.push_back(std::move(step_pos_tensor));
      for (auto &kv : past_kv) {
        step_inputs.push_back(std::move(kv));
      }

      auto step_out =
          sess_gpt_->Run({}, gpt_input_names_ptr_.data(), step_inputs.data(),
                         step_inputs.size(), gpt_output_names_ptr_.data(),
                         gpt_output_names_ptr_.size());

      // Sample from logits at position 0
      const float *step_logits = step_out[0].GetTensorData<float>();
      auto step_shape = step_out[0].GetTensorTypeAndShapeInfo().GetShape();
      int32_t V = static_cast<int32_t>(step_shape[2]);

      std::vector<float> adj_logits(step_logits, step_logits + V);
      ApplyRepetitionPenalty(adj_logits.data(), V, generated_tokens,
                             repetition_penalty);

      int32_t sampled =
          SampleTopP(adj_logits.data(), V, temperature, top_p, &rng_);
      generated_tokens.push_back(static_cast<int64_t>(sampled));

      // Update KV cache
      past_kv.clear();
      for (int32_t i = 1; i <= 2 * kNumLayers; ++i) {
        past_kv.push_back(std::move(step_out[i]));
      }

      if (sampled == kEosId) {
        generated_tokens.pop_back();  // remove EOS
        if (config_.debug) SHERPA_ONNX_LOGE("Hit EOS at step %d. Ending decode loop.", step);
        break;
      }
    }

    // Strip the audio offset to get raw MioCodec content indices
    std::vector<int64_t> audio_tokens;
    audio_tokens.reserve(generated_tokens.size());
    for (int64_t t : generated_tokens) {
      if (t >= kAudioOffset && t < kAudioOffset + kAudioVocabSize) {
        audio_tokens.push_back(t - kAudioOffset);
      }
    }
    return audio_tokens;
  }

  std::vector<float> DecodeMiocodec(const int64_t *content_indices,
                                    int32_t num_tokens,
                                    const float *global_embedding,
                                    int32_t global_dim) {
    if (config_.debug) {
      SHERPA_ONNX_LOGE("DecodeMiocodec starts with %d tokens and %d global_dim.", num_tokens, global_dim);
    }
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // content_indices: (1, T)
    std::array<int64_t, 2> idx_shape = {1, num_tokens};
    Ort::Value idx_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<int64_t *>(content_indices),
        static_cast<size_t>(num_tokens), idx_shape.data(), idx_shape.size());

    // global_embedding: (1, C)
    std::array<int64_t, 2> g_shape = {1, global_dim};
    Ort::Value g_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(global_embedding),
        static_cast<size_t>(global_dim), g_shape.data(), g_shape.size());

    std::array<Ort::Value, 2> inputs = {std::move(idx_tensor),
                                        std::move(g_tensor)};

    // Explicitly define input names to match the order of `inputs`!
    const char *input_names[] = {"content_indices", "global_embedding"};

    try {
      auto out = sess_miocodec_dec_->Run(
          {}, input_names, inputs.data(), inputs.size(),
          dec_output_names_ptr_.data(), dec_output_names_ptr_.size());

      const float *wav = out[0].GetTensorData<float>();
      auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
      int64_t n = 1;
      for (auto d : shape) n *= d;
      
      if (config_.debug) {
        SHERPA_ONNX_LOGE("DecodeMiocodec finished with %lld output elements.", (long long)n);
      }
      
      // Apply Perth watermarker if available
      if (sess_perth_watermarker_) {
        // Reshape to [1, n] for Perth watermarker
        std::vector<int64_t> audio_shape = {1, n};
        Ort::Value audio_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float *>(wav), n,
            audio_shape.data(), audio_shape.size());
            
        // Run Perth watermarker
        const char* watermarker_input_name = "audio_values";
        const char* watermarker_output_name = "watermarked_audio_values";
        Ort::RunOptions run_options{nullptr};
        auto watermarked_outputs = sess_perth_watermarker_->Run(
            run_options, &watermarker_input_name, &audio_tensor, 1,
            &watermarker_output_name, 1);
            
        const float *wm_wav = watermarked_outputs[0].GetTensorData<float>();
        auto wm_shape = watermarked_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t wm_n = 1;
        for (auto d : wm_shape) wm_n *= d;
        
        return std::vector<float>(wm_wav, wm_wav + wm_n);
      }

      return std::vector<float>(wav, wav + n);
    } catch (const Ort::Exception& e) {
      SHERPA_ONNX_LOGE("ONNX Runtime Exception in DecodeMiocodec: %s", e.what());
      return {};
    }
  }

 private:
  // ---------------------------------------------------------------------------
  // Initializer helpers
  // ---------------------------------------------------------------------------

  void Init(const char *gpt_data, size_t gpt_size, const char *campplus_data,
            size_t campplus_size, const char *enc_data, size_t enc_size,
            const char *dec_data, size_t dec_size, const char *emb_data,
            size_t emb_size) {
    // Create ONNX sessions
    sess_gpt_ = std::make_unique<Ort::Session>(
        env_, gpt_data, gpt_size, sess_opts_);
    sess_campplus_ = std::make_unique<Ort::Session>(
        env_, campplus_data, campplus_size, sess_opts_);
    sess_miocodec_enc_ = std::make_unique<Ort::Session>(
        env_, enc_data, enc_size, sess_opts_);
    sess_miocodec_dec_ = std::make_unique<Ort::Session>(
        env_, dec_data, dec_size, sess_opts_);

    // Cache input/output name pointers for all sessions
    GetInputNames(sess_gpt_.get(), &gpt_input_names_, &gpt_input_names_ptr_);
    GetOutputNames(sess_gpt_.get(), &gpt_output_names_, &gpt_output_names_ptr_);

    GetInputNames(sess_campplus_.get(), &campplus_input_names_,
                  &campplus_input_names_ptr_);
    GetOutputNames(sess_campplus_.get(), &campplus_output_names_,
                   &campplus_output_names_ptr_);

    GetInputNames(sess_miocodec_enc_.get(), &enc_input_names_,
                  &enc_input_names_ptr_);
    GetOutputNames(sess_miocodec_enc_.get(), &enc_output_names_,
                   &enc_output_names_ptr_);

    GetInputNames(sess_miocodec_dec_.get(), &dec_input_names_,
                  &dec_input_names_ptr_);
    GetOutputNames(sess_miocodec_dec_.get(), &dec_output_names_,
                   &dec_output_names_ptr_);

    {
      Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
      Ort::ModelMetadata meta_data = sess_miocodec_dec_->GetModelMetadata();
      SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(sample_rate_, "sample_rate", 24000);
    }

    // Parse embeddings.npz
    std::unordered_map<std::string, NpzArray> arrays;
    if (!ParseNpz(reinterpret_cast<const uint8_t *>(emb_data), emb_size,
                  &arrays)) {
      SHERPA_ONNX_LOGE("Failed to parse embeddings.npz");
      SHERPA_ONNX_EXIT(-1);
    }

    auto load_array = [&](const std::string &name,
                          std::vector<float> *out) {
      auto it = arrays.find(name);
      if (it == arrays.end()) {
        SHERPA_ONNX_LOGE("embeddings.npz: missing array '%s'", name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      *out = std::move(it->second.data);
    };

    load_array("tok_emb_weight", &tok_emb_weight_);
    load_array("spk_proj_weight", &spk_proj_weight_);
    load_array("spk_proj_bias", &spk_proj_bias_);

    if (!config_.miocodec_llama.perth_watermarker.empty()) {
#ifdef _WIN32
      sess_perth_watermarker_ = std::make_unique<Ort::Session>(
          env_, StrToWstr(config_.miocodec_llama.perth_watermarker).c_str(), sess_opts_);
#else
      sess_perth_watermarker_ = std::make_unique<Ort::Session>(
          env_, config_.miocodec_llama.perth_watermarker.c_str(), sess_opts_);
#endif
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Perth watermarker loaded: %s", config_.miocodec_llama.perth_watermarker.c_str());
      }
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE(
          "MioCodec-LLaMA: tok_emb=%zu, spk_proj_weight=%zu, "
          "spk_proj_bias=%zu",
          tok_emb_weight_.size(), spk_proj_weight_.size(),
          spk_proj_bias_.size());
    }
  }

  // ---------------------------------------------------------------------------
  // Log-mel fbank computation (80 bins, 16 kHz, 25ms window, 10ms shift)
  // ---------------------------------------------------------------------------
  // Returns a flat vector of shape (num_frames * 80).
  static std::vector<float> ComputeFbank(const float *samples, int32_t n,
                                         int32_t sample_rate) {
    const int32_t frame_len = static_cast<int32_t>(0.025f * sample_rate);  // 400
    const int32_t frame_shift = static_cast<int32_t>(0.010f * sample_rate); // 160
    const int32_t num_bins = 80;
    const float kPreEmphCoeff = 0.97f;

    // Compute number of frames using standard formula
    int32_t num_frames = (n >= frame_len)
                             ? (1 + (n - frame_len) / frame_shift)
                             : 0;
    if (num_frames == 0) return {};

    // Simple Hamming window
    std::vector<float> window(frame_len);
    for (int32_t i = 0; i < frame_len; ++i) {
      window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (frame_len - 1));
    }

    // Mel filter bank boundaries
    const float kMelLow = 0.0f;
    const float kMelHigh = 2595.0f * std::log10(1.0f + sample_rate / 2.0f / 700.0f);
    auto hz2mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel2hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

    int32_t fft_len = 512;
    while (fft_len < frame_len) fft_len <<= 1;

    std::vector<float> mel_points(num_bins + 2);
    for (int32_t i = 0; i < num_bins + 2; ++i) {
      float mel = kMelLow + (kMelHigh - kMelLow) * i / (num_bins + 1);
      mel_points[i] = mel2hz(mel) * fft_len / sample_rate;
    }

    // Build mel filterbank matrix (num_bins, fft_len/2+1)
    int32_t fft_half = fft_len / 2 + 1;
    std::vector<float> filterbank(num_bins * fft_half, 0.0f);
    for (int32_t m = 0; m < num_bins; ++m) {
      float left = mel_points[m];
      float center = mel_points[m + 1];
      float right = mel_points[m + 2];
      for (int32_t k = 0; k < fft_half; ++k) {
        float fk = static_cast<float>(k);
        if (fk >= left && fk <= center) {
          filterbank[m * fft_half + k] = (fk - left) / (center - left + 1e-8f);
        } else if (fk > center && fk <= right) {
          filterbank[m * fft_half + k] = (right - fk) / (right - center + 1e-8f);
        }
      }
    }

    std::vector<float> result(static_cast<size_t>(num_frames) * num_bins, 0.0f);

    for (int32_t t = 0; t < num_frames; ++t) {
      int32_t start = t * frame_shift;

      // Pre-emphasis + windowing
      std::vector<float> frame(fft_len, 0.0f);
      float prev = (start > 0) ? samples[start - 1] : 0.0f;
      for (int32_t i = 0; i < frame_len && start + i < n; ++i) {
        float cur = samples[start + i];
        frame[i] = window[i] * (cur - kPreEmphCoeff * prev);
        prev = cur;
      }

      // FFT (DFT for simplicity — real FFT not available in stdlib)
      // We compute power spectrum directly via DFT over fft_half bins.
      std::vector<float> power(fft_half, 0.0f);
      for (int32_t k = 0; k < fft_half; ++k) {
        float re = 0, im = 0;
        for (int32_t i = 0; i < fft_len; ++i) {
          float angle = -2.0f * M_PI * k * i / fft_len;
          re += frame[i] * std::cos(angle);
          im += frame[i] * std::sin(angle);
        }
        power[k] = re * re + im * im;
      }

      // Apply mel filterbank and take log
      for (int32_t m = 0; m < num_bins; ++m) {
        float energy = 0;
        for (int32_t k = 0; k < fft_half; ++k) {
          energy += filterbank[m * fft_half + k] * power[k];
        }
        result[t * num_bins + m] = std::log(energy + 1e-10f);
      }
    }

    return result;
  }

  // Apply repetition penalty to logits for already-generated tokens
  static void ApplyRepetitionPenalty(float *logits, int32_t vocab_size,
                                     const std::vector<int64_t> &seen,
                                     float penalty) {
    if (std::abs(penalty - 1.0f) < 1e-6f) return;
    for (int64_t tok : seen) {
      if (tok >= 0 && tok < vocab_size) {
        if (logits[tok] > 0) {
          logits[tok] /= penalty;
        } else {
          logits[tok] *= penalty;
        }
      }
    }
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::mt19937 rng_;

  std::unique_ptr<Ort::Session> sess_gpt_;
  std::unique_ptr<Ort::Session> sess_campplus_;
  std::unique_ptr<Ort::Session> sess_miocodec_enc_;
  std::unique_ptr<Ort::Session> sess_miocodec_dec_;
  std::unique_ptr<Ort::Session> sess_perth_watermarker_;

  std::vector<std::string> gpt_input_names_;
  std::vector<const char *> gpt_input_names_ptr_;
  std::vector<std::string> gpt_output_names_;
  std::vector<const char *> gpt_output_names_ptr_;

  std::vector<std::string> campplus_input_names_;
  std::vector<const char *> campplus_input_names_ptr_;
  std::vector<std::string> campplus_output_names_;
  std::vector<const char *> campplus_output_names_ptr_;

  std::vector<std::string> enc_input_names_;
  std::vector<const char *> enc_input_names_ptr_;
  std::vector<std::string> enc_output_names_;
  std::vector<const char *> enc_output_names_ptr_;

  std::vector<std::string> dec_input_names_;
  std::vector<const char *> dec_input_names_ptr_;
  std::vector<std::string> dec_output_names_;
  std::vector<const char *> dec_output_names_ptr_;

  // Embedding weights loaded from embeddings.npz
  std::vector<float> tok_emb_weight_;    // (vocab_size, hidden_dim)
  std::vector<float> spk_proj_weight_;   // (hidden_dim, 192)
  std::vector<float> spk_proj_bias_;     // (hidden_dim,)

  int32_t sample_rate_ = 24000;
};

// ---------------------------------------------------------------------------
// Public interface forwarding to Impl
// ---------------------------------------------------------------------------

OfflineTtsMiocodecLlamaModel::OfflineTtsMiocodecLlamaModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsMiocodecLlamaModel::OfflineTtsMiocodecLlamaModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsMiocodecLlamaModel::~OfflineTtsMiocodecLlamaModel() = default;

int32_t OfflineTtsMiocodecLlamaModel::SampleRate() const {
  return impl_->SampleRate();
}

std::vector<float> OfflineTtsMiocodecLlamaModel::ExtractSpeakerEmbedding(
    const float *audio_samples, int32_t audio_len) {
  return impl_->ExtractSpeakerEmbedding(audio_samples, audio_len);
}

MiocodecFeatures OfflineTtsMiocodecLlamaModel::ExtractMiocodecFeatures(
    const float *audio_samples, int32_t audio_len) {
  return impl_->ExtractMiocodecFeatures(audio_samples, audio_len);
}

std::vector<int64_t> OfflineTtsMiocodecLlamaModel::GenerateAudioTokens(
    const std::vector<int64_t> &phoneme_ids,
    const std::vector<float> &speaker_embedding, int32_t language_id,
    float temperature, float top_p, int32_t max_tokens,
    float repetition_penalty) {
  return impl_->GenerateAudioTokens(phoneme_ids, speaker_embedding,
                                    language_id, temperature, top_p,
                                    max_tokens, repetition_penalty);
}

std::vector<float> OfflineTtsMiocodecLlamaModel::DecodeMiocodec(
    const int64_t *content_indices, int32_t num_tokens,
    const float *global_embedding, int32_t global_dim) {
  return impl_->DecodeMiocodec(content_indices, num_tokens, global_embedding,
                               global_dim);
}

#if __ANDROID_API__ >= 9
template OfflineTtsMiocodecLlamaModel::OfflineTtsMiocodecLlamaModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsMiocodecLlamaModel::OfflineTtsMiocodecLlamaModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
