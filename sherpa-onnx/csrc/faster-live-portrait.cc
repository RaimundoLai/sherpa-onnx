// Copyright (c) 2026 Xiaomi Corporation

#include "sherpa-onnx/csrc/faster-live-portrait.h"

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {
namespace {

FasterLivePortraitTensorType ToTensorType(
    ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return FasterLivePortraitTensorType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return FasterLivePortraitTensorType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return FasterLivePortraitTensorType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return FasterLivePortraitTensorType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return FasterLivePortraitTensorType::kUint8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return FasterLivePortraitTensorType::kBool;
    default:
      throw std::invalid_argument("Unsupported FasterLivePortrait tensor type");
  }
}

size_t TypeSize(FasterLivePortraitTensorType type) {
  switch (type) {
    case FasterLivePortraitTensorType::kFloat32:
      return sizeof(float);
    case FasterLivePortraitTensorType::kFloat16:
      return sizeof(uint16_t);
    case FasterLivePortraitTensorType::kInt64:
      return sizeof(int64_t);
    case FasterLivePortraitTensorType::kInt32:
      return sizeof(int32_t);
    case FasterLivePortraitTensorType::kUint8:
    case FasterLivePortraitTensorType::kBool:
      return sizeof(uint8_t);
    default:
      throw std::invalid_argument("Unsupported FasterLivePortrait tensor type");
  }
}

size_t ShapeElementCount(const std::vector<int64_t> &shape) {
  size_t count = 1;
  for (int64_t dimension : shape) {
    if (dimension <= 0) {
      throw std::invalid_argument(
          "FasterLivePortrait tensor dimensions must be positive");
    }
    if (static_cast<uint64_t>(dimension) >
        std::numeric_limits<size_t>::max() / count) {
      throw std::invalid_argument("FasterLivePortrait tensor is too large");
    }
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

void ValidateShape(const std::vector<int64_t> &expected,
                   const std::vector<int64_t> &actual,
                   const std::string &model_name,
                   const std::string &tensor_name) {
  if (expected.size() != actual.size()) {
    throw std::invalid_argument("Tensor rank mismatch for model '" +
                                model_name + "', input '" + tensor_name +
                                "'");
  }
  for (size_t i = 0; i != expected.size(); ++i) {
    // ONNX uses -1 for a dynamic dimension.  Other negative dimensions are
    // invalid for a runtime input.
    if (expected[i] < -1 || (expected[i] >= 0 && expected[i] != actual[i])) {
      throw std::invalid_argument("Tensor shape mismatch for model '" +
                                  model_name + "', input '" + tensor_name +
                                  "'");
    }
  }
}

Ort::Value MakeTensor(const FasterLivePortraitTensorInput &input,
                      const Ort::MemoryInfo &memory_info) {
  auto shape = input.shape;
  const size_t count = ShapeElementCount(shape);
  if (count != input.element_count) {
    throw std::invalid_argument("FasterLivePortrait tensor element count does "
                                "not match its shape");
  }
  if (count != 0 && input.data == nullptr) {
    throw std::invalid_argument("FasterLivePortrait tensor data is null");
  }

  switch (input.type) {
    case FasterLivePortraitTensorType::kFloat32:
      return Ort::Value::CreateTensor<float>(
          memory_info, const_cast<float *>(
                           static_cast<const float *>(input.data)),
          count, shape.data(), shape.size());
    case FasterLivePortraitTensorType::kFloat16:
      return Ort::Value::CreateTensor(
          memory_info, const_cast<void *>(input.data),
          count * sizeof(uint16_t), shape.data(), shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    case FasterLivePortraitTensorType::kInt64:
      return Ort::Value::CreateTensor<int64_t>(
          memory_info, const_cast<int64_t *>(
                           static_cast<const int64_t *>(input.data)),
          count, shape.data(), shape.size());
    case FasterLivePortraitTensorType::kInt32:
      return Ort::Value::CreateTensor<int32_t>(
          memory_info, const_cast<int32_t *>(
                           static_cast<const int32_t *>(input.data)),
          count, shape.data(), shape.size());
    case FasterLivePortraitTensorType::kUint8:
      return Ort::Value::CreateTensor<uint8_t>(
          memory_info, const_cast<uint8_t *>(
                           static_cast<const uint8_t *>(input.data)),
          count, shape.data(), shape.size());
    case FasterLivePortraitTensorType::kBool:
      return Ort::Value::CreateTensor(
          memory_info, const_cast<void *>(input.data), count * sizeof(uint8_t),
          shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    default:
      throw std::invalid_argument("Unsupported FasterLivePortrait input type");
  }
}

template <typename T>
void CopyTensorBytes(const Ort::Value &value, std::vector<uint8_t> *data) {
  const size_t count =
      value.GetTensorTypeAndShapeInfo().GetElementCount();
  data->resize(count * sizeof(T));
  std::memcpy(data->data(), value.GetTensorData<T>(), data->size());
}

}  // namespace

class FasterLivePortraitModelSet::Impl {
 public:
  struct ModelSession {
    FasterLivePortraitModelInfo info;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<const char *> input_names_ptr;
    std::vector<std::string> output_names;
    std::vector<const char *> output_names_ptr;
  };

  explicit Impl(const FasterLivePortraitConfig &config) {
    if (config.models.empty()) {
      throw std::invalid_argument(
          "At least one FasterLivePortrait ONNX model is required");
    }

    const int32_t num_threads = config.num_threads > 0 ? config.num_threads : 1;
    const std::string provider = config.provider.empty() ? "cpu" : config.provider;

    models.reserve(config.models.size());
    infos.reserve(config.models.size());
    model_indices.reserve(config.models.size());
    for (const auto &model_config : config.models) {
      if (model_config.name.empty()) {
        throw std::invalid_argument(
            "FasterLivePortrait model name must not be empty");
      }
      if (model_config.path.empty()) {
        throw std::invalid_argument("FasterLivePortrait model '" +
                                    model_config.name + "' has no path");
      }
      if (model_indices.find(model_config.name) != model_indices.end()) {
        throw std::invalid_argument("Duplicate FasterLivePortrait model name: " +
                                    model_config.name);
      }

      auto model = std::make_unique<ModelSession>();
      model->info.name = model_config.name;
      model->info.path = model_config.path;
      // CoreML's provider factory may use ONNX Runtime's default logger while
      // appending the execution provider. Initialize the shared ORT
      // environment before creating SessionOptions so CoreML does not access
      // an unregistered logger.
      auto &ort_env = GetOrtEnv();
      auto options = GetSessionOptions(num_threads, provider);
      model->session = std::make_unique<Ort::Session>(
          ort_env, model_config.path.c_str(), options);
      GetInputNames(model->session.get(), &model->input_names,
                    &model->input_names_ptr);
      GetOutputNames(model->session.get(), &model->output_names,
                     &model->output_names_ptr);

      for (size_t i = 0; i != model->input_names.size(); ++i) {
        auto type_info = model->session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        FasterLivePortraitTensorInfo info;
        info.name = model->input_names[i];
        info.type = ToTensorType(tensor_info.GetElementType());
        info.shape = tensor_info.GetShape();
        model->info.inputs.push_back(std::move(info));
      }
      for (size_t i = 0; i != model->output_names.size(); ++i) {
        auto type_info = model->session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        FasterLivePortraitTensorInfo info;
        info.name = model->output_names[i];
        info.type = ToTensorType(tensor_info.GetElementType());
        info.shape = tensor_info.GetShape();
        model->info.outputs.push_back(std::move(info));
      }

      model_indices.emplace(model_config.name, models.size());
      infos.push_back(model->info);
      models.push_back(std::move(model));
    }
  }

  std::vector<std::unique_ptr<ModelSession>> models;
  std::vector<FasterLivePortraitModelInfo> infos;
  std::unordered_map<std::string, size_t> model_indices;
};

FasterLivePortraitModelSet::FasterLivePortraitModelSet(
    const FasterLivePortraitConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

FasterLivePortraitModelSet::~FasterLivePortraitModelSet() = default;

const std::vector<FasterLivePortraitModelInfo> &
FasterLivePortraitModelSet::GetModelInfos() const {
  static const std::vector<FasterLivePortraitModelInfo> kEmpty;
  if (!impl_) return kEmpty;
  return impl_->infos;
}

std::vector<FasterLivePortraitTensorOutput> FasterLivePortraitModelSet::Run(
    const std::string &model_name,
    const std::vector<FasterLivePortraitTensorInput> &inputs) const {
  if (!impl_) throw std::invalid_argument("FasterLivePortrait is not initialized");
  auto index = impl_->model_indices.find(model_name);
  if (index == impl_->model_indices.end()) {
    throw std::invalid_argument("Unknown FasterLivePortrait model: " +
                                model_name);
  }
  const auto &model = *impl_->models[index->second];
  if (inputs.size() != model.info.inputs.size()) {
    throw std::invalid_argument("Model '" + model_name + "' expects " +
                                std::to_string(model.info.inputs.size()) +
                                " inputs, got " +
                                std::to_string(inputs.size()));
  }

  std::vector<const FasterLivePortraitTensorInput *> ordered(
      model.info.inputs.size(), nullptr);
  for (size_t i = 0; i != inputs.size(); ++i) {
    size_t target = i;
    if (!inputs[i].name.empty()) {
      target = model.info.inputs.size();
      for (size_t j = 0; j != model.info.inputs.size(); ++j) {
        if (model.info.inputs[j].name == inputs[i].name) {
          target = j;
          break;
        }
      }
    }
    if (target >= ordered.size()) {
      throw std::invalid_argument("Unknown input '" + inputs[i].name +
                                  "' for model '" + model_name + "'");
    }
    if (ordered[target] != nullptr) {
      throw std::invalid_argument("Duplicate input '" +
                                  model.info.inputs[target].name +
                                  "' for model '" + model_name + "'");
    }
    const auto &expected = model.info.inputs[target];
    if (inputs[i].type != expected.type) {
      throw std::invalid_argument("Tensor type mismatch for model '" +
                                  model_name + "', input '" + expected.name +
                                  "'");
    }
    ValidateShape(expected.shape, inputs[i].shape, model_name, expected.name);
    if (ShapeElementCount(inputs[i].shape) != inputs[i].element_count) {
      throw std::invalid_argument("Tensor element count mismatch for input '" +
                                  expected.name + "'");
    }
    ordered[target] = &inputs[i];
  }
  for (size_t i = 0; i != ordered.size(); ++i) {
    if (ordered[i] == nullptr) {
      throw std::invalid_argument("Missing input '" + model.info.inputs[i].name +
                                  "' for model '" + model_name + "'");
    }
  }

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(ordered.size());
  for (const auto *input : ordered) {
    ort_inputs.push_back(MakeTensor(*input, memory_info));
  }

  auto ort_outputs = model.session->Run(
      {}, model.input_names_ptr.data(), ort_inputs.data(), ort_inputs.size(),
      model.output_names_ptr.data(), model.output_names_ptr.size());

  std::vector<FasterLivePortraitTensorOutput> outputs;
  outputs.reserve(ort_outputs.size());
  for (size_t i = 0; i != ort_outputs.size(); ++i) {
    const auto &value = ort_outputs[i];
    if (!value.IsTensor()) {
      throw std::invalid_argument("Non-tensor output from model '" +
                                  model_name + "'");
    }
    auto tensor_info = value.GetTensorTypeAndShapeInfo();
    FasterLivePortraitTensorOutput output;
    output.name = model.output_names[i];
    output.type = ToTensorType(tensor_info.GetElementType());
    output.shape = tensor_info.GetShape();
    switch (output.type) {
      case FasterLivePortraitTensorType::kFloat32:
        CopyTensorBytes<float>(value, &output.data);
        break;
      case FasterLivePortraitTensorType::kFloat16:
        CopyTensorBytes<uint16_t>(value, &output.data);
        break;
      case FasterLivePortraitTensorType::kInt64:
        CopyTensorBytes<int64_t>(value, &output.data);
        break;
      case FasterLivePortraitTensorType::kInt32:
        CopyTensorBytes<int32_t>(value, &output.data);
        break;
      case FasterLivePortraitTensorType::kUint8:
        CopyTensorBytes<uint8_t>(value, &output.data);
        break;
      case FasterLivePortraitTensorType::kBool:
        CopyTensorBytes<bool>(value, &output.data);
        break;
      default:
        throw std::invalid_argument("Unsupported output from model '" +
                                    model_name + "'");
    }
    outputs.push_back(std::move(output));
  }
  return outputs;
}

}  // namespace sherpa_onnx
