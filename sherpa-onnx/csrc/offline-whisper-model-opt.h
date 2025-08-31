// sherpa-onnx/csrc/offline-whisper-model.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_OPT_H_
#define SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_OPT_H_

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/spoken-language-identification.h"

namespace sherpa_onnx {

class OfflineWhisperModelOpt {
 public:
  explicit OfflineWhisperModelOpt(const OfflineModelConfig &config);

  explicit OfflineWhisperModelOpt(
      const SpokenLanguageIdentificationConfig &config);

  template <typename Manager>
  OfflineWhisperModelOpt(Manager *mgr, const OfflineModelConfig &config);

  template <typename Manager>
  OfflineWhisperModelOpt(Manager *mgr,
                         const SpokenLanguageIdentificationConfig &config);

  ~OfflineWhisperModelOpt();

  /** Run the encoder model.
   *
   * @param features  A tensor of shape (N, C, T). It is changed in-place.
   *                  C is 80 and T is 3000.
   *
   * @return Return a pair containing:
   *  - n_layer_cross_k: A 4-D tensor of shape
   *                     (n_text_layer, N, n_audio_ctx, n_text_state)
   *  - n_layer_cross_v: A 4-D tensor of shape
   *                     (n_text_layer, N, n_audio_ctx, n_text_state)
   */
  std::pair<Ort::Value, Ort::Value> ForwardEncoder(Ort::Value features) const;

  /** Run the decoder model.
   *
   * @param tokens A int64 tensor of shape (N, 1)
   * @param n_layer_self_k_cache  A 4-D tensor of shape
   *                              (n_text_layer, N, n_text_ctx, n_text_state).
   * @param n_layer_self_v_cache  A 4-D tensor of shape
   *                              (n_text_layer, N, n_text_ctx, n_text_state).
   * @param n_layer_cross_k       A 4-D tensor of shape
   *                              (n_text_layer, N, n_audio_ctx, n_text_state).
   * @param n_layer_cross_v       A 4-D tensor of shape
   *                              (n_text_layer, N, n_audio_ctx, n_text_state).
   * @param offset A int64 tensor of shape (B,), init with 0
   * @param attention_mask A int32 tensor of (B,), init with 1
   * @param sel A bool tensor of (B, n_text_ctx, 1), init with B * [true, false,
   *  false, false, ...]
   *
   * @return Return a tuple containing 7 tensors:
   *
   *  - logits A 3-D tensor of shape (N, 1, vocab_size)
   *  - out_n_layer_self_k_cache A 4-D tensor of shape
   *                              (n_text_layer, N, n_text_ctx, n_text_state).
   *  - out_n_layer_self_v_cache A 4-D tensor of shape
   *                              (n_text_layer, N, n_text_ctx, n_text_state).
   *  - token_next A 2-D tensor of shape (N, 1)
   *  - offset_next A int64 tensor of shape (B,)
   *  - mask_next A int32 tensor of shape (B,)
   *  - sel_next A bool tensor of shape (B, n_text_ctx, 1)
   */
  std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
             Ort::Value, Ort::Value>
  ForwardDecoder(Ort::Value tokens, Ort::Value n_layer_self_k_cache,
                 Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
                 Ort::Value n_layer_cross_v, Ort::Value offset,
                 Ort::Value attention_mask, Ort::Value sel) const;

  std::vector<int32_t> DetectLanguage(Ort::Value &cross_k,   // NOLINT
                                      Ort::Value &cross_v);  // NOLINT

  void ForwardEncoderWithBinding(Ort::Value features);

  std::tuple<Ort::Value> ForwardDecoderWithBinding(Ort::Value tokens);

  std::tuple<Ort::Value> ForwardDecoderWithBinding();

  std::vector<int32_t> DetectLanguageWithBinding();

  /** Reset the step counter for a new sequence generation */
  void ResetStep();

  /** Return the initial self kv cache in a pair
   *  - n_layer_self_k_cache A 4-D tensor of shape
   *                         (n_text_layer, N, n_audio_ctx, n_text_state).
   *  - n_layer_self_v_cache A 4-D tensor of shape
   *                         (n_text_layer, N, n_audio_ctx, n_text_state).
   */
  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache(
      const int32_t batch_size = 1) const;
  const std::vector<int64_t> &GetInitialTokens() const;
  const std::vector<int32_t> &GetAllLanguageIDs() const;
  const std::unordered_map<std::string, int32_t> &GetLang2ID() const;
  const std::unordered_map<int32_t, std::string> &GetID2Lang() const;

  /** Return an allocator for allocating memory
   */
  OrtAllocator *Allocator() const;

  int32_t NoTimeStampsToken() const;
  int32_t EOT() const;
  int32_t SOT() const;
  int32_t TextCtx() const;
  int32_t VocabSize() const;
  int32_t FeatureDim() const;
  int32_t Translate() const;
  bool IsMultiLingual() const;

  static void NormalizeFeatures(float *features, int32_t num_frames,
                                int32_t feat_dim);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_WHISPER_MODEL_OPT_H_
