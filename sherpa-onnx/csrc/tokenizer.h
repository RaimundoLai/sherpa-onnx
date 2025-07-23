#ifndef SHERPA_ONNX_CSRC_TOKENIZER_H_
#define SHERPA_ONNX_CSRC_TOKENIZER_H_

#include <memory>
#include <string>
#include <vector>

namespace sherpa_onnx {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual std::vector<int64_t> Tokenize(const std::string &text, const std::string &lang) const = 0;
};

std::unique_ptr<Tokenizer> CreateTokenizer(const std::string &model_path,
               const std::unordered_map<std::string, int32_t>& token2id);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TOKENIZER_H_