// sherpa-onnx/csrc/piper-phonemize-lexicon.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"

#include <codecvt>
#include <fstream>
#include <locale>
#include <map>
#include <mutex>  // NOLINT
#include <sstream>
#include <string>
#include <strstream>
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
#include "sherpa-onnx/csrc/tokenizer-phonemize.h"

namespace sherpa_onnx {


// Encode a single char32_t to UTF-8 string. For debugging only
static std::string ToString(char32_t cp) {
  std::string result;

  if (cp <= 0x7F) {
    result += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    result += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    result += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0x10FFFF) {
    result += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    SHERPA_ONNX_LOGE("Invalid Unicode code point: %d",
                     static_cast<int32_t>(cp));
  }

  return result;
}

void CallPhonemizeEspeak(const std::string &text,
                         piper::eSpeakPhonemeConfig &config,  // NOLINT
                         std::vector<std::vector<piper::Phoneme>> *phonemes) {
  static std::mutex espeak_mutex;

  std::lock_guard<std::mutex> lock(espeak_mutex);

  // keep multi threads from calling into piper::phonemize_eSpeak
  piper::phonemize_eSpeak(text, config, *phonemes);
}

static std::unordered_map<char32_t, int32_t> ReadTokens(std::istream &is) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  std::unordered_map<char32_t, int32_t> token2id;

  std::string line;

  std::string sym;
  std::u32string s;
  int32_t id = 0;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    iss >> sym;
    if (iss.eof()) {
      id = atoi(sym.c_str());
      sym = " ";
    } else {
      iss >> id;
    }

    // eat the trailing \r\n on windows
    iss >> std::ws;
    if (!iss.eof()) {
      SHERPA_ONNX_LOGE("Error when reading tokens: %s", line.c_str());
      exit(-1);
    }

    s = conv.from_bytes(sym);
    if (s.size() != 1) {
      // for tokens.txt from coqui-ai/TTS, the last token is <BLNK>
      if (s.size() == 6 && s[0] == '<' && s[1] == 'B' && s[2] == 'L' &&
          s[3] == 'N' && s[4] == 'K' && s[5] == '>') {
        continue;
      }

      SHERPA_ONNX_LOGE("Error when reading tokens at Line %s. size: %d",
                       line.c_str(), static_cast<int32_t>(s.size()));
      exit(-1);
    }

    char32_t c = s[0];

    if (token2id.count(c)) {
      SHERPA_ONNX_LOGE("Duplicated token %s. Line %s. Existing ID: %d",
                       sym.c_str(), line.c_str(), token2id.at(c));
      exit(-1);
    }

    token2id.insert({c, id});
  }

  return token2id;
}


static std::vector<int64_t> PiperIdsToVitsIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<int64_t> &phoneme_ids) {
  int32_t pad = token2id.at(U'_');
  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  std::vector<int64_t> ans;
  ans.reserve(phoneme_ids.size() * 2 + 2);

  ans.push_back(bos);
  for (int64_t p_id : phoneme_ids) {
    ans.push_back(p_id);
    ans.push_back(pad);
  }
  ans.push_back(eos);

  return ans;
}


static std::vector<std::vector<int64_t>> PiperPhonemesToIdsMatcha(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes, bool use_eos_bos,
    int32_t max_token_len = 400) {
  // We set max_token_len to 400 here to fix
  // https://github.com/k2-fsa/sherpa-onnx/issues/2666
  std::vector<std::vector<int64_t>> ans;
  std::vector<int64_t> current;

  int32_t bos = token2id.at(U'^');
  int32_t eos = token2id.at(U'$');

  if (use_eos_bos) {
    current.push_back(bos);
  }

  for (auto p : phonemes) {
    if (token2id.count(p)) {
      current.push_back(token2id.at(p));
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }

    if (current.size() > max_token_len + 1) {
      if (use_eos_bos) {
        current.push_back(eos);
      }

      ans.push_back(std::move(current));

      if (use_eos_bos) {
        current.push_back(bos);
      }
    }
  }  // for (auto p : phonemes)

  if (!current.empty()) {
    if (use_eos_bos) {
      if (current.size() > 1) {
        current.push_back(eos);

        ans.push_back(std::move(current));
      }
    } else {
      ans.push_back(std::move(current));
    }
  }

  return ans;
}

static std::vector<std::vector<int64_t>> PiperPhonemesToIdsKokoroOrKitten(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<piper::Phoneme> &phonemes, int32_t max_len) {
  std::vector<std::vector<int64_t>> ans;
  if (phoneme_ids.empty()) {
    return ans;
  }

  std::vector<int64_t> current;
  current.reserve(phoneme_ids.size());

  current.push_back(0); // BOS ID

  for (auto p : phonemes) {
    // SHERPA_ONNX_LOGE("%d %s", static_cast<int32_t>(p), ToString(p).c_str());
    if (token2id.count(p)) {
      if (current.size() > max_len - 1) {
        current.push_back(0);
        ans.push_back(std::move(current));

        current.reserve(phonemes.size());
        current.push_back(0);
      }

      current.push_back(token2id.at(p));
      if (p == '.') {
        current.push_back(token2id.at(' '));
      }
    } else {
      SHERPA_ONNX_LOGE("Skip unknown phonemes. Unicode codepoint: \\U+%04x.",
                       static_cast<uint32_t>(p));
    }

    current.push_back(p_id);
  }

  current.push_back(0); // EOS ID
  ans.push_back(std::move(current));
  return ans;
}
std::string char32_to_utf8_string(char32_t c) {
  std::string result;
  if (c < 0x80) {
      // 1-byte sequence (ASCII)
      result += static_cast<char>(c);
  } else if (c < 0x800) {
      // 2-byte sequence
      result += static_cast<char>(0xC0 | (c >> 6));
      result += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
      // 3-byte sequence
      result += static_cast<char>(0xE0 | (c >> 12));
      result += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      result += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x110000) {
      // 4-byte sequence
      result += static_cast<char>(0xF0 | (c >> 18));
      result += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
      result += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
      result += static_cast<char>(0x80 | (c & 0x3F));
  } else {
      throw std::runtime_error("Invalid char32_t code point for UTF-8 conversion");
  }
  return result;
}
static std::vector<int64_t> CoquiIdsToCoquiIds(
    const std::unordered_map<char32_t, int32_t> &token2id,
    const std::vector<int64_t> &phoneme_ids,
    const OfflineTtsVitsModelMetaData &vits_meta_data) {
  int32_t use_eos_bos = vits_meta_data.use_eos_bos;
  int32_t bos_id = vits_meta_data.bos_id;
  int32_t eos_id = vits_meta_data.eos_id;
  int32_t blank_id = vits_meta_data.blank_id;
  int32_t add_blank = vits_meta_data.add_blank;
  int32_t comma_id = token2id.at(',');

  std::vector<int64_t> ans;
  if (add_blank) {
    ans.reserve(phoneme_ids.size() * 2 + 3);
  } else {
    ans.reserve(phoneme_ids.size() + 2);
  }

  if (use_eos_bos) {
    ans.push_back(bos_id);
  }

  if (add_blank) {
    ans.push_back(blank_id);
    for (auto p_id : phoneme_ids) {
      ans.push_back(p_id);
      ans.push_back(blank_id);
    }
  } else {
    ans.insert(ans.end(), phoneme_ids.begin(), phoneme_ids.end());
  }

  // add a comma at the end of a sentence so that we can have a longer pause.
  ans.push_back(comma_id);

  if (use_eos_bos) {
    ans.push_back(eos_id);
  }

  return ans;
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
  const std::string &tokens,
  const OfflineTtsVitsModelMetaData &vits_meta_data)
  : vits_meta_data_(vits_meta_data) {
{
  std::ifstream is(tokens);
  token2id_ = ReadTokensForPiper(is);
}
}


PiperPhonemizeLexicon::PiperPhonemizeLexicon(
  const std::string &tokens,
  const OfflineTtsMatchaModelMetaData &matcha_meta_data)
  : matcha_meta_data_(matcha_meta_data), is_matcha_(true) {
{
  std::ifstream is(tokens);
  token2id_ = ReadTokensForPiper(is);
}
}
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsVitsModelMetaData &vits_meta_data)
    : vits_meta_data_(vits_meta_data) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokensForPiper(is);
    for (const auto& pair : token2id_) {
      new_token2id_[char32_to_utf8_string(pair.first)] = pair.second;
    }
  }
  tokenizer_ =
      CreateTokenizer(g2p_model, new_token2id_);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &g2p_model,
   const std::string &tokens,
    const OfflineTtsVitsModelMetaData &vits_meta_data)
    : vits_meta_data_(vits_meta_data) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokensForPiper(is);
    for (const auto& pair : token2id_) {
      new_token2id_[char32_to_utf8_string(pair.first)] = pair.second;
    }
  }
  tokenizer_ =
      CreateTokenizer(g2p_model, new_token2id_);
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &g2p_model, 
    const std::string &tokens,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data)
    : matcha_meta_data_(matcha_meta_data), is_matcha_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokensForPiper(is);
    for (const auto& pair : token2id_) {
      new_token2id_[char32_to_utf8_string(pair.first)] = pair.second;
    }
  }
  tokenizer_ = CreateTokenizer(g2p_model, new_token2id_);
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &g2p_model, 
    const std::string &tokens,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data)
    : kokoro_meta_data_(kokoro_meta_data), is_kokoro_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokensForPiper(is);
    for (const auto& pair : token2id_) {
      new_token2id_[char32_to_utf8_string(pair.first)] = pair.second;
    }
  }
  tokenizer_ = CreateTokenizer(g2p_model, new_token2id_);
}

PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kitten_meta_data)
    : kitten_meta_data_(kitten_meta_data), is_kitten_(true) {
  {
    std::ifstream is(tokens);
    token2id_ = ReadTokens(is);
  }

  InitEspeak(data_dir);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data)
    : matcha_meta_data_(matcha_meta_data), is_matcha_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokensForPiper(is);
    for (const auto& pair : token2id_) {
      new_token2id_[char32_to_utf8_string(pair.first)] = pair.second;
    }
  }
  tokenizer_ = CreateTokenizer(g2p_model, new_token2id_);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data)
    : kokoro_meta_data_(kokoro_meta_data), is_kokoro_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokensForPiper(is);
    for (const auto& pair : token2id_) {
      new_token2id_[char32_to_utf8_string(pair.first)] = pair.second;
    }
  }
  tokenizer_ = CreateTokenizer(g2p_model, new_token2id_);
}

template <typename Manager>
PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    Manager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kitten_meta_data)
    : kitten_meta_data_(kitten_meta_data), is_kitten_(true) {
  {
    auto buf = ReadFile(mgr, tokens);
    std::istrstream is(buf.data(), buf.size());
    token2id_ = ReadTokens(is);
  }

  // We should copy the directory of espeak-ng-data from the asset to
  // some internal or external storage and then pass the directory to
  // data_dir.
  InitEspeak(data_dir);
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIds(
    const std::string &text, const std::string &voice /*= ""*/) const {
  if (is_matcha_) {
    return ConvertTextToTokenIdsMatcha(text, voice);
  } else if (is_kokoro_) {
    return ConvertTextToTokenIdsKokoroOrKitten(
        token2id_, kokoro_meta_data_.max_token_len, text, voice);
  } else if (is_kitten_) {
    return ConvertTextToTokenIdsKokoroOrKitten(
        token2id_, kitten_meta_data_.max_token_len, text, voice);
  } else {
    return ConvertTextToTokenIdsVits(text, voice);
  }
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsMatcha(
    const std::string &text, const std::string &voice /*= ""*/) const {
  std::vector<int64_t> phoneme_ids = tokenizer_->Tokenize(text, voice);

  std::vector<int64_t> final_ids =
      PiperIdsToMatchaIds(token2id_, phoneme_ids, matcha_meta_data_.use_eos_bos);

  std::vector<TokenIDs> ans;
  ans.emplace_back(std::move(final_ids));
  return ans;
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsKokoro(
    const std::string &text, const std::string &voice /*= ""*/) const {
  std::vector<int64_t> phoneme_ids = tokenizer_->Tokenize(text, voice);

  for (const auto &p : phonemes) {
    auto phoneme_ids =
        PiperPhonemesToIdsMatcha(token2id_, p, matcha_meta_data_.use_eos_bos);

    for (auto &ids : phoneme_ids) {
      ans.emplace_back(std::move(ids));
    }
  }

  return ans;
}

std::vector<TokenIDs> ConvertTextToTokenIdsKokoroOrKitten(
    const std::unordered_map<char32_t, int32_t> &token2id,
    int32_t max_token_len, const std::string &text,
    const std::string &voice /*= ""*/) {
  piper::eSpeakPhonemeConfig config;

  auto segmented_ids =
      PiperIdsToKokoroIds(phoneme_ids, kokoro_meta_data_.max_token_len);

  std::vector<TokenIDs> ans;
  for (auto &ids : segmented_ids) {
    ans.emplace_back(std::move(ids));
  }

  return ans;
}

std::vector<TokenIDs> PiperPhonemizeLexicon::ConvertTextToTokenIdsVits(
    const std::string &text, const std::string &voice /*= ""*/) const {
  std::vector<int64_t> phoneme_ids = tokenizer_->Tokenize(text, voice);

  std::vector<TokenIDs> ans;
  std::vector<int64_t> final_ids;

  if (vits_meta_data_.is_piper || vits_meta_data_.is_icefall) {
    final_ids = PiperIdsToVitsIds(token2id_, phoneme_ids);
  } else if (vits_meta_data_.is_coqui) {
    final_ids = CoquiIdsToCoquiIds(token2id_, phoneme_ids, vits_meta_data_);
  } else {
    SHERPA_ONNX_LOGE("Unsupported model");
    exit(-1);
  }

  ans.emplace_back(std::move(final_ids));
  return ans;
}

#if __ANDROID_API__ >= 9
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &g2p_model,
     const std::string &tokens,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    AAssetManager *mgr, const std::string &tokens, const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kokoro_meta_data);
#endif

#if __OHOS__
template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsVitsModelMetaData &vits_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsMatchaModelMetaData &matcha_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &g2p_model,
    const std::string &tokens,
    const OfflineTtsKokoroModelMetaData &kokoro_meta_data);

template PiperPhonemizeLexicon::PiperPhonemizeLexicon(
    NativeResourceManager *mgr, const std::string &tokens,
    const std::string &data_dir,
    const OfflineTtsKittenModelMetaData &kokoro_meta_data);
#endif

}  // namespace sherpa_onnx