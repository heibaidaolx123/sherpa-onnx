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
    printf("Initialized OfflineRecognizerWhisperImplOpt.\n");
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

    if (config_.model_config.io_binding) {
      printf("Using IO binding for whisper encoder\n");
      model_->ForwardEncoderWithBinding(std::move(mel));
      printf("Using IO binding for whisper decoder\n");
      int32_t batch_size = n;
      std::array<int64_t, 1> fake_kv_shape{1};
      Ort::Value fake_k =
          Ort::Value::CreateTensor(memory_info, &batch_size, 1,
                                   fake_kv_shape.data(), fake_kv_shape.size());
      Ort::Value fake_v =
          Ort::Value::CreateTensor(memory_info, &batch_size, 1,
                                   fake_kv_shape.data(), fake_kv_shape.size());

      auto result_vec = decoder_->Decode(
          std::move(fake_k), std::move(fake_v),
          *std::max_element(num_frames_vec.begin(), num_frames_vec.end()));
      for (int i = 0; i < n; ++i) {
        auto s = ss[i];
        auto r = Convert(result_vec[i], symbol_table_);
        s->SetResult(r);
      }
      return;
    }

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

      auto result_vec = decoder_->Decode(
          std::move(cross_kv_batched.first), std::move(cross_kv_batched.second),
          *std::max_element(num_frames_vec.begin(), num_frames_vec.end()));
      for (int i = 0; i < n; ++i) {
        auto s = ss[i];
        auto r = Convert(result_vec[i], symbol_table_);
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
    std::vector<OfflineStream *> streams = {s};
    DecodeStreams(streams.data(), streams.size());
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
