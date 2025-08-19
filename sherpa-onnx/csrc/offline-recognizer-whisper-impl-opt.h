// sherpa-onnx/csrc/offline-recognizer-whisper-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_OPT_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_OPT_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-whisper-decoder.h"
#include "sherpa-onnx/csrc/offline-whisper-greedy-search-decoder-opt.h"
#include "sherpa-onnx/csrc/offline-whisper-model-opt.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineRecognizerWhisperImplOpt : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerWhisperImplOpt(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineWhisperModelOpt>(config.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerWhisperImplOpt(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineWhisperModelOpt>(mgr,
                                                        config.model_config)) {
    Init();
  }

  void Init() {
    // tokens.txt from whisper is base64 encoded, so we need to decode it
    symbol_table_.ApplyBase64Decode();

    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineWhisperGreedySearchDecoderOpt>(
          config_.model_config.whisper, model_.get());
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search is supported at present for whisper. Given %s",
          config_.decoding_method.c_str());
      exit(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    WhisperTag tag;
    tag.dim = model_->FeatureDim();
    return std::make_unique<OfflineStream>(tag);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // // batch decoding is not implemented yet
    // for (int32_t i = 0; i != n; ++i) {
    //   DecodeStream(ss[i]);
    // }
    if (n == 1) {
      DecodeStream(ss[0]);
      return;
    }
    decoder_->SetConfig(config_.model_config.whisper);

    int32_t max_num_frames = 3000;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::array<int64_t, 3> shape{n, max_num_frames, feat_dim};
    Ort::Value mel = Ort::Value::CreateTensor<float>(
        model_->Allocator(), shape.data(), shape.size());
    float *p_mel = mel.GetTensorMutableData<float>();

    std::vector<int32_t> num_frames_vec(n);
    for (int32_t i = 0; i < n; ++i) {
      OfflineStream *s = ss[i];
      std::vector<float> f = s->GetFrames();
      int32_t num_frames = f.size() / feat_dim;
      num_frames_vec[i] = num_frames;
      model_->NormalizeFeatures(f.data(), num_frames, feat_dim);
      std::copy(f.data(), f.data() + num_frames * feat_dim,
                p_mel + i * max_num_frames * feat_dim);
      std::fill_n(p_mel + (max_num_frames * i + num_frames) * feat_dim,
                  (max_num_frames - num_frames) * feat_dim, 0);
    }
    mel = Transpose12(model_->Allocator(), &mel);

    try {
      auto cross_kv_batched = model_->ForwardEncoder(std::move(mel));
      auto kshape =
          cross_kv_batched.first.GetTensorTypeAndShapeInfo().GetShape();
      int32_t n_text_layer = kshape[0];
      int32_t n_bacth = kshape[1];
      if (n_bacth != n) {
        SHERPA_ONNX_LOGE(
            "Expected batch size %d, but got %d. Please check your input.", n,
            n_bacth);
        return;
      }
      int32_t n_audio_ctx = kshape[2];
      int32_t n_text_state = kshape[3];

      float *p_k_batched = cross_kv_batched.first.GetTensorMutableData<float>();
      float *p_v_batched =
          cross_kv_batched.second.GetTensorMutableData<float>();

      std::array<int64_t, 4> shape{n_text_layer, 1, n_audio_ctx, n_text_state};
      std::vector<float> k_buf(n_text_layer * n_audio_ctx * n_text_state);
      std::vector<float> v_buf(n_text_layer * n_audio_ctx * n_text_state);

      for (int32_t i = 0; i < n; ++i) {
        // for (int32_t i = n - 1; i >= 0; --i) {
        auto s = ss[i];
        auto num_frames = num_frames_vec[i];

        Ort::Value n_layer_cross_k = Ort::Value::CreateTensor<float>(
            memory_info, k_buf.data(), k_buf.size(), shape.data(),
            shape.size());
        Ort::Value n_layer_cross_v = Ort::Value::CreateTensor<float>(
            memory_info, v_buf.data(), v_buf.size(), shape.data(),
            shape.size());
        // Ort::Value n_layer_cross_k = Ort::Value::CreateTensor<float>(
        //     model_->Allocator(), shape.data(), shape.size());
        // Ort::Value n_layer_cross_v = Ort::Value::CreateTensor<float>(
        //     model_->Allocator(), shape.data(), shape.size());

        for (int32_t l = 0; l < n_text_layer; ++l) {
          float *p_k_l_src =
              p_k_batched + (l * n + i) * n_audio_ctx * n_text_state;
          float *p_v_l_src =
              p_v_batched + (l * n + i) * n_audio_ctx * n_text_state;
          float *p_k_l_dst = k_buf.data() + l * n_audio_ctx * n_text_state;
          float *p_v_l_dst = v_buf.data() + l * n_audio_ctx * n_text_state;
          // float *p_k_l_dst = n_layer_cross_k.GetTensorMutableData<float>() +
          //                    l * n_audio_ctx * n_text_state;
          // float *p_v_l_dst = n_layer_cross_v.GetTensorMutableData<float>() +
          //                    l * n_audio_ctx * n_text_state;

          std::copy(p_k_l_src, p_k_l_src + n_audio_ctx * n_text_state,
                    p_k_l_dst);
          std::copy(p_v_l_src, p_v_l_src + n_audio_ctx * n_text_state,
                    p_v_l_dst);
        }
        auto results = decoder_->Decode(std::move(n_layer_cross_k),
                                        std::move(n_layer_cross_v), num_frames);

        auto r = Convert(results[0], symbol_table_);
        s->SetResult(r);
      }
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE("\n\nCaught exception:\n\n%s", ex.what());
      return;
    }
  }

  void SetConfig(const OfflineRecognizerConfig &config) override {
    config_.model_config.whisper = config.model_config.whisper;
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeStream(OfflineStream *s) const {
    decoder_->SetConfig(config_.model_config.whisper);

    int32_t max_num_frames = 3000;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();
    int32_t num_frames = f.size() / feat_dim;

    // we use 50 here so that there will be some zero tail paddings
    if (num_frames >= max_num_frames - 50) {
      SHERPA_ONNX_LOGE(
          "Only waves less than 30 seconds are supported. We process only the "
          "first 30 seconds and discard the remaining data");
      num_frames = max_num_frames - 50;
    }

    model_->NormalizeFeatures(f.data(), num_frames, feat_dim);

    // note that 1000 is an experience-value.
    // You can replace 1000 by other values, say, 100.
    //
    // Since we have removed the 30 seconds constraint, we need
    // tail_padding_frames so that whisper is able to detect the eot token.
    int32_t tail_padding_frames = 1000;

    if (config_.model_config.whisper.tail_paddings > 0) {
      tail_padding_frames = config_.model_config.whisper.tail_paddings;
    }

    int32_t actual_frames =
        std::min(num_frames + tail_padding_frames, max_num_frames);

    std::array<int64_t, 3> shape{1, actual_frames, feat_dim};

    Ort::Value mel = Ort::Value::CreateTensor<float>(
        model_->Allocator(), shape.data(), shape.size());

    float *p_mel = mel.GetTensorMutableData<float>();
    std::copy(f.data(), f.data() + num_frames * feat_dim, p_mel);

    std::fill_n(p_mel + num_frames * feat_dim,
                (actual_frames - num_frames) * feat_dim, 0);

    mel = Transpose12(model_->Allocator(), &mel);

    try {
      auto cross_kv = model_->ForwardEncoder(std::move(mel));

      auto results = decoder_->Decode(std::move(cross_kv.first),
                                      std::move(cross_kv.second), num_frames);

      auto r = Convert(results[0], symbol_table_);
      s->SetResult(r);
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE(
          "\n\nCaught exception:\n\n%s\n\nReturn an empty result. Number of "
          "input frames: %d, Current tail "
          "paddings: %d. If you see a lot of such exceptions, please consider "
          "using a larger --whisper-tail-paddings",
          ex.what(), num_frames, tail_padding_frames);
      return;
    }
  }

 private:
  OfflineRecognitionResult Convert(const OfflineWhisperDecoderResult &src,
                                   const SymbolTable &sym_table) const {
    OfflineRecognitionResult r;
    r.tokens.reserve(src.tokens.size());

    std::string text;
    for (auto i : src.tokens) {
      if (!sym_table.Contains(i)) {
        continue;
      }

      std::string s = sym_table[i];
      s = ApplyInverseTextNormalization(s);
      s = ApplyHomophoneReplacer(std::move(s));

      text += s;
      r.tokens.push_back(s);
    }

    r.text = text;
    r.lang = src.lang;

    return r;
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineWhisperModelOpt> model_;
  std::unique_ptr<OfflineWhisperDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_WHISPER_IMPL_H_
