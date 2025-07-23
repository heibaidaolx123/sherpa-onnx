// sherpa-onnx/csrc/speaker-embedding-extractor-nemo-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_NEMO_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_NEMO_IMPL_H_
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-nemo-model.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorNeMoImpl : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorNeMoImpl(
      const SpeakerEmbeddingExtractorConfig &config)
      : model_(config) {}

  template <typename Manager>
  SpeakerEmbeddingExtractorNeMoImpl(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : model_(mgr, config) {}

  int32_t Dim() const override { return model_.GetMetaData().output_dim; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    FeatureExtractorConfig feat_config;
    const auto &meta_data = model_.GetMetaData();
    feat_config.sampling_rate = meta_data.sample_rate;
    feat_config.feature_dim = meta_data.feat_dim;
    feat_config.normalize_samples = true;
    feat_config.snip_edges = true;
    feat_config.frame_shift_ms = meta_data.window_stride_ms;
    feat_config.frame_length_ms = meta_data.window_size_ms;
    feat_config.low_freq = 0;
    feat_config.is_librosa = true;
    feat_config.remove_dc_offset = false;
    feat_config.window_type = meta_data.window_type;

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
      if (meta_data.feature_normalize_type == "per_feature") {
        NormalizePerFeature(features.data(), num_frames, feat_dim);
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

    if (num_frames % 16 != 0) {
      int32_t pad = 16 - num_frames % 16;
      features.resize((num_frames + pad) * feat_dim);
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{1, num_frames, feat_dim};
    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, features.data(), features.size(),
                                 x_shape.data(), x_shape.size());

    x = Transpose12(model_.Allocator(), &x);

    int64_t x_lens = num_frames;
    std::array<int64_t, 1> x_lens_shape{1};
    Ort::Value x_lens_tensor = Ort::Value::CreateTensor(
        memory_info, &x_lens, 1, x_lens_shape.data(), x_lens_shape.size());

    Ort::Value embedding =
        model_.Compute(std::move(x), std::move(x_lens_tensor));
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
    int feat_dim = ss[0]->FeatureDim();
    bool apply_cmvn = false;
    const auto &meta_data = model_.GetMetaData();
    if (!meta_data.feature_normalize_type.empty()) {
      if (meta_data.feature_normalize_type == "per_feature") {
        apply_cmvn = true;
      }
    }
    std::vector<std::vector<float>> results;
    results.resize(ss.size());

    std::vector<OnlineStream *> filtered_ss;
    std::vector<int> filtered_ss_index;
    int max_num_frames = 0;
    std::vector<int64_t> num_frames_per_stream;

    for (size_t i = 0; i < ss.size(); ++i) {
      OnlineStream *s = ss[i];
      int num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
      if (num_frames <= 0) {
        results[i] = {};
        continue;
      }
      if (num_frames > max_num_frames) {
        max_num_frames = num_frames;
      }
      filtered_ss.push_back(ss[i]);
      filtered_ss_index.push_back(i);
      num_frames_per_stream.push_back(num_frames);
    }

    if (filtered_ss.empty()) {
      return results;
    }

    if (max_num_frames % 16 != 0) {
      int32_t pad = 16 - max_num_frames % 16;
      max_num_frames += pad;
    }

    int batch_size = filtered_ss.size();
    std::vector<float> buffer(batch_size * max_num_frames * feat_dim, 0.0f);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    for (int i = 0; i < batch_size; ++i) {
      OnlineStream *s = filtered_ss[i];
      int num_frames = num_frames_per_stream[i];
      std::vector<float> features =
          s->GetFrames(s->GetNumProcessedFrames(), num_frames);

      s->GetNumProcessedFrames() += num_frames;

      int32_t feat_dim_tmp = features.size() / num_frames;
      if (feat_dim_tmp != feat_dim) {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "Feature dimension mismatch: expected %d, got %d for stream %d",
            feat_dim, feat_dim_tmp, filtered_ss_index[i]);
#else
        SHERPA_ONNX_LOGE(
            "Feature dimension mismatch: expected %d, got %d for stream %d",
            feat_dim, feat_dim_tmp, filtered_ss_index[i]);
#endif
        exit(-1);
      }
      if (apply_cmvn) {
        NormalizePerFeature(features.data(), num_frames, feat_dim_tmp);
      }

      std::copy(features.begin(), features.end(),
                buffer.data() + i * max_num_frames * feat_dim);
    }

    std::array<int64_t, 3> x_shape{batch_size, max_num_frames, feat_dim};
    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, buffer.data(), buffer.size(),
                                 x_shape.data(), x_shape.size());

    x = Transpose12(model_.Allocator(), &x);

    std::array<int64_t, 1> x_lens_shape{batch_size};
    Ort::Value x_lens_tensor = Ort::Value::CreateTensor(
        memory_info, num_frames_per_stream.data(), num_frames_per_stream.size(),
        x_lens_shape.data(), x_lens_shape.size());

    Ort::Value embedding =
        model_.Compute(std::move(x), std::move(x_lens_tensor));
    std::vector<int64_t> embedding_shape =
        embedding.GetTensorTypeAndShapeInfo().GetShape();

    for (int i = 0; i < batch_size; ++i) {
      std::vector<float> &results_i = results[filtered_ss_index[i]];
      results_i.resize(embedding_shape[1]);
      std::copy(embedding.GetTensorData<float>() + i * embedding_shape[1],
                embedding.GetTensorData<float>() + (i + 1) * embedding_shape[1],
                results_i.begin());
    }

    return results;
  }

 private:
  void NormalizePerFeature(float *p, int32_t num_frames,
                           int32_t feat_dim) const {
    auto m = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        p, num_frames, feat_dim);

    auto EX = m.colwise().mean();
    auto EX2 = m.array().pow(2).colwise().sum() / num_frames;
    auto variance = EX2 - EX.array().pow(2);
    auto stddev = variance.array().sqrt();

    m = (m.rowwise() - EX).array().rowwise() / (stddev.array() + 1e-5);
  }

 private:
  SpeakerEmbeddingExtractorNeMoModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_NEMO_IMPL_H_
