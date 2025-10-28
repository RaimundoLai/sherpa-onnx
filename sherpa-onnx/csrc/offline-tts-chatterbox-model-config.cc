// sherpa-onnx/csrc/offline-tts-chatterbox-model-config.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-chatterbox-model-config.h"

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineTtsChatterboxModelConfig::Register(ParseOptions *po) {
  po->Register("chatterbox-speech-encoder", &speech_encoder,
               "Path to speech_encoder.onnx from chatterbox");
  po->Register("chatterbox-embed-tokens", &embed_tokens,
               "Path to embed_tokens.onnx from chatterbox");
  po->Register("chatterbox-language-model", &language_model,
               "Path to language_model.onnx from chatterbox");
  po->Register("chatterbox-conditional-decoder", &conditional_decoder,
               "Path to conditional_decoder.onnx from chatterbox");
  po->Register("chatterbox-tokenizer", &tokenizer,
               "Path to tokenizer.json from chatterbox");
  po->Register("chatterbox-lang", &lang,
               "Language of the model. e.g., en. Currently ignored.");
  po->Register("chatterbox-lexicon", &lexicon,
               "Path to lexicon.txt from chatterbox");
  po->Register("chatterbox-cangjie-dict", &cangjie_dict,
               "Path to cangjie_dict.txt from chatterbox");
}

bool OfflineTtsChatterboxModelConfig::Validate() const {
  if (speech_encoder.empty() || embed_tokens.empty() ||
      language_model.empty() || conditional_decoder.empty() ||
      tokenizer.empty()) {
    SHERPA_ONNX_LOGE(
        "Please provide all chatterbox model and tokenizer files.");
    return false;
  }

  if (!FileExists(speech_encoder)) {
    SHERPA_ONNX_LOGE("chatterbox speech encoder file not found: %s",
                     speech_encoder.c_str());
    return false;
  }

  if (!FileExists(embed_tokens)) {
    SHERPA_ONNX_LOGE("chatterbox embed tokens file not found: %s",
                     embed_tokens.c_str());
    return false;
  }

  if (!FileExists(language_model)) {
    SHERPA_ONNX_LOGE("chatterbox language model file not found: %s",
                     language_model.c_str());
    return false;
  }

  if (!FileExists(conditional_decoder)) {
    SHERPA_ONNX_LOGE("chatterbox conditional decoder file not found: %s",
                     conditional_decoder.c_str());
    return false;
  }

  if (!FileExists(tokenizer)) {
    SHERPA_ONNX_LOGE("chatterbox tokenizer file not found: %s",
                     tokenizer.c_str());
    return false;
  }

  return true;
}

std::string OfflineTtsChatterboxModelConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineTtsChatterboxModelConfig(";
  os << "speech_encoder=\"" << speech_encoder << "\", ";
  os << "embed_tokens=\"" << embed_tokens << "\", ";
  os << "language_model=\"" << language_model << "\", ";
  os << "conditional_decoder=\"" << conditional_decoder << "\", ";
  os << "tokenizer=\"" << tokenizer << "\", ";
  os << "lexicon=\"" << lexicon << "\", ";
  os << "cangjie_dict=\"" << cangjie_dict << "\", ";
  os << "lang=\"" << lang << "\")";
  return os.str();
}

}  // namespace sherpa_onnx
