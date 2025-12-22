#ifndef SHERPA_ONNX_CSRC_TOKENIZER_PHONEMIZE_H_
#define SHERPA_ONNX_CSRC_TOKENIZER_PHONEMIZE_H_

#include <memory>
#include <string>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OnnxG2p {
 public:

  OnnxG2p(const std::string &model, const std::string &tokenizer_path,
          const std::string &provider, bool debug);

  ~OnnxG2p();


  std::vector<int32_t> Tokenize(const std::string &text, const std::string &lang) const;
  public:
  const SymbolTable &GetSymbolTable() const;
 private:
  class OnnxG2pImpl;
  std::unique_ptr<OnnxG2pImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TOKENIZER_PHONEMIZE_H_