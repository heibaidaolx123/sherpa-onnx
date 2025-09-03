// sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-model.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorGeneralImpl
    : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorGeneralImpl(
      const SpeakerEmbeddingExtractorConfig &config)
      : model_(config) {}

  template <typename Manager>
  SpeakerEmbeddingExtractorGeneralImpl(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : model_(mgr, config) {}

  int32_t Dim() const override { return model_.GetMetaData().output_dim; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    FeatureExtractorConfig feat_config;
    const auto &meta_data = model_.GetMetaData();
    feat_config.sampling_rate = meta_data.sample_rate;
    feat_config.normalize_samples = meta_data.normalize_samples;

    return std::make_unique<OnlineStream>(feat_config);
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() < s->NumFramesReady();
  }

  std::vector<float> Compute(OnlineStream *s) const override {
    int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
    if (num_frames <= 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %{public}d",
          num_frames);
#else
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %d",
          num_frames);
#endif
      return {};
    }

    std::vector<float> features =
        s->GetFrames(s->GetNumProcessedFrames(), num_frames);

    s->GetNumProcessedFrames() += num_frames;

    int32_t feat_dim = features.size() / num_frames;

    const auto &meta_data = model_.GetMetaData();
    if (!meta_data.feature_normalize_type.empty()) {
      if (meta_data.feature_normalize_type == "global-mean") {
        SubtractGlobalMean(features.data(), num_frames, feat_dim);
      } else {
#if __OHOS__
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %{public}s",
                         meta_data.feature_normalize_type.c_str());
#else
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %s",
                         meta_data.feature_normalize_type.c_str());
#endif
        exit(-1);
      }
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{1, num_frames, feat_dim};
    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, features.data(), features.size(),
                                 x_shape.data(), x_shape.size());
    Ort::Value embedding = model_.Compute(std::move(x));
    std::vector<int64_t> embedding_shape =
        embedding.GetTensorTypeAndShapeInfo().GetShape();

    std::vector<float> ans(embedding_shape[1]);
    std::copy(embedding.GetTensorData<float>(),
              embedding.GetTensorData<float>() + ans.size(), ans.begin());

    return ans;
  }

  std::vector<std::vector<float>> ComputeMultiple(
      std::vector<OnlineStream *> ss) const override {
    if (ss.empty()) {
      SHERPA_ONNX_LOGE("No streams provided for processing");
      return {};
    }
    int32_t batch_size = ss.size();
    int32_t max_num_frames = 0;
    std::vector<int32_t> num_frames_per_stream;
    for (auto &s : ss) {
      int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
      if (num_frames <= 0) {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "Please make sure IsReady(s) returns true. num_frames: %{public}d",
            num_frames);
#else
        SHERPA_ONNX_LOGE(
            "Please make sure IsReady(s) returns true. num_frames: %d",
            num_frames);
#endif
        exit(-1);
      }
      num_frames_per_stream.push_back(num_frames);
      if (num_frames > max_num_frames) {
        max_num_frames = num_frames;
      }
    }

    const auto &meta_data = model_.GetMetaData();
    int32_t feat_dim = ss[0]->FeatureDim();
    std::vector<float> batch_features(batch_size * max_num_frames * feat_dim,
                                      0.0f);
    for (int i = 0; i < ss.size(); ++i) {
      auto &s = ss[i];
      int32_t num_frames = num_frames_per_stream[i];
      std::vector<float> features =
          s->GetFrames(s->GetNumProcessedFrames(), num_frames);
      if (!meta_data.feature_normalize_type.empty()) {
        if (meta_data.feature_normalize_type == "global-mean") {
          SubtractGlobalMean(features.data(), num_frames, feat_dim);
        } else {
#if __OHOS__
          SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %{public}s",
                           meta_data.feature_normalize_type.c_str());
#else
          SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %s",
                           meta_data.feature_normalize_type.c_str());
#endif
          exit(-1);
        }
      }
      s->GetNumProcessedFrames() += num_frames;
      int32_t feat_dim_fthis_stream = features.size() / num_frames;
      if (feat_dim_fthis_stream != feat_dim) {
        SHERPA_ONNX_LOGE("Inconsistent feature dimension: %d vs %d",
                         feat_dim_fthis_stream, feat_dim);
        exit(-1);
      }

      std::copy(features.begin(), features.end(),
                batch_features.data() + i * max_num_frames * feat_dim);
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{batch_size, max_num_frames, feat_dim};
    Ort::Value x = Ort::Value::CreateTensor(memory_info, batch_features.data(),
                                            batch_features.size(),
                                            x_shape.data(), x_shape.size());
    Ort::Value embedding = model_.Compute(std::move(x));
    std::vector<int64_t> embedding_shape =
        embedding.GetTensorTypeAndShapeInfo().GetShape();

    std::vector<std::vector<float>> results(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      std::vector<float> &results_i = results[i];
      results_i.resize(embedding_shape[1]);
      std::copy(embedding.GetTensorData<float>() + i * embedding_shape[1],
                embedding.GetTensorData<float>() + (i + 1) * embedding_shape[1],
                results_i.begin());
    }
    return results;
  }

 private:
  void SubtractGlobalMean(float *p, int32_t num_frames,
                          int32_t feat_dim) const {
    auto m = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        p, num_frames, feat_dim);

    m = m.rowwise() - m.colwise().mean();
  }

 private:
  SpeakerEmbeddingExtractorModel model_;
  const SpeakerEmbeddingExtractorConfig config_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
