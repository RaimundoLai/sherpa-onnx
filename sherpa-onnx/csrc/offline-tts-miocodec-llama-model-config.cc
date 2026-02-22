// sherpa-onnx/csrc/offline-tts-miocodec-llama-model-config.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-miocodec-llama-model-config.h"

#include <sstream>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void OfflineTtsMiocodecLlamaModelConfig::Register(ParseOptions *po) {
  po->Register("miocodec-llama-model", &model,
               "Path to gpt_linacodec.onnx (KV-cache GPT model) for "
               "MioCodec-LLaMA TTS");
  po->Register("miocodec-llama-campplus-model", &campplus_model,
               "Path to campplus.onnx for speaker embedding extraction");
  po->Register("miocodec-llama-encoder", &miocodec_encoder,
               "Path to miocodec_encoder.onnx");
  po->Register("miocodec-llama-decoder", &miocodec_decoder,
               "Path to miocodec_decoder.onnx");
  po->Register("miocodec-llama-embeddings", &embeddings,
               "Path to embeddings.npz containing token embedding and "
               "speaker projection weights");
  po->Register("miocodec-llama-tokens", &tokens,
               "Path to phoneme_vocab.json");
  po->Register("miocodec-llama-lexicon", &lexicon,
               "Path to lexicon file(s) for G2P. Multiple files separated "
               "by ','");
  po->Register("miocodec-llama-g2p-model", &g2p_model,
               "Path to G2P ONNX model for phonemization");
  po->Register("miocodec-llama-temperature", &temperature,
               "Sampling temperature for audio token generation");
  po->Register("miocodec-llama-top-p", &top_p,
               "Top-p (nucleus) sampling threshold");
  po->Register("miocodec-llama-max-tokens", &max_tokens,
               "Maximum number of audio tokens to generate");
  po->Register("miocodec-llama-repetition-penalty", &repetition_penalty,
               "Repetition penalty for audio token generation (1.0 = no "
               "penalty)");
  po->Register("miocodec-llama-perth-watermarker", &perth_watermarker,
               "Path to Perth watermarker ONNX model for audio watermarking (optional)");
}

bool OfflineTtsMiocodecLlamaModelConfig::Validate() const {
  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --miocodec-llama-model");
    return false;
  }
  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("--miocodec-llama-model: '%s' does not exist",
                     model.c_str());
    return false;
  }

  if (campplus_model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --miocodec-llama-campplus-model");
    return false;
  }
  if (!FileExists(campplus_model)) {
    SHERPA_ONNX_LOGE(
        "--miocodec-llama-campplus-model: '%s' does not exist",
        campplus_model.c_str());
    return false;
  }

  if (miocodec_encoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --miocodec-llama-encoder");
    return false;
  }
  if (!FileExists(miocodec_encoder)) {
    SHERPA_ONNX_LOGE("--miocodec-llama-encoder: '%s' does not exist",
                     miocodec_encoder.c_str());
    return false;
  }

  if (miocodec_decoder.empty()) {
    SHERPA_ONNX_LOGE("Please provide --miocodec-llama-decoder");
    return false;
  }
  if (!FileExists(miocodec_decoder)) {
    SHERPA_ONNX_LOGE("--miocodec-llama-decoder: '%s' does not exist",
                     miocodec_decoder.c_str());
    return false;
  }

  if (embeddings.empty()) {
    SHERPA_ONNX_LOGE("Please provide --miocodec-llama-embeddings");
    return false;
  }
  if (!FileExists(embeddings)) {
    SHERPA_ONNX_LOGE("--miocodec-llama-embeddings: '%s' does not exist",
                     embeddings.c_str());
    return false;
  }

  if (tokens.empty()) {
    SHERPA_ONNX_LOGE("Please provide --miocodec-llama-tokens");
    return false;
  }
  if (!FileExists(tokens)) {
    SHERPA_ONNX_LOGE("--miocodec-llama-tokens: '%s' does not exist",
                     tokens.c_str());
    return false;
  }

  if (!lexicon.empty()) {
    std::vector<std::string> files;
    SplitStringToVector(lexicon, ",", false, &files);
    for (const auto &f : files) {
      if (!FileExists(f)) {
        SHERPA_ONNX_LOGE(
            "lexicon '%s' does not exist. Please re-check "
            "--miocodec-llama-lexicon",
            f.c_str());
        return false;
      }
    }
  }

  if (!g2p_model.empty() && !FileExists(g2p_model)) {
    SHERPA_ONNX_LOGE("--miocodec-llama-g2p-model: '%s' does not exist",
                     g2p_model.c_str());
    return false;
  }

  // perth_watermarker is optional, but validate if provided
  if (!perth_watermarker.empty() && !FileExists(perth_watermarker)) {
    SHERPA_ONNX_LOGE("miocodec-llama perth watermarker file not found: %s",
                     perth_watermarker.c_str());
    return false;
  }

  return true;
}

std::string OfflineTtsMiocodecLlamaModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineTtsMiocodecLlamaModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "campplus_model=\"" << campplus_model << "\", ";
  os << "miocodec_encoder=\"" << miocodec_encoder << "\", ";
  os << "miocodec_decoder=\"" << miocodec_decoder << "\", ";
  os << "embeddings=\"" << embeddings << "\", ";
  os << "tokens=\"" << tokens << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "g2p_model=\"" << g2p_model << "\", ";
  os << "temperature=" << temperature << ", ";
  os << "top_p=" << top_p << ", ";
  os << "max_tokens=" << max_tokens << ", ";
  os << "repetition_penalty=" << repetition_penalty << ", ";
  os << "perth_watermarker=\"" << perth_watermarker << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
