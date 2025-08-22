// sherpa-onnx/csrc/offline-whisper-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-greedy-search-decoder-opt.h"

#include <algorithm>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

void OfflineWhisperGreedySearchDecoderOpt::SetConfig(
    const OfflineWhisperModelConfig &config) {
  config_ = config;
}

std::vector<OfflineWhisperDecoderResult>
OfflineWhisperGreedySearchDecoderOpt::Decode(Ort::Value cross_k,
                                             Ort::Value cross_v,
                                             int32_t num_feature_frames) {
  auto shape = cross_k.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() == 1) {
    return DecodeIOBinding(std::move(cross_k), num_feature_frames);
  } else {
    return DecodeOrig(std::move(cross_k), std::move(cross_v),
                      num_feature_frames);
  }
}

std::vector<OfflineWhisperDecoderResult>
OfflineWhisperGreedySearchDecoderOpt::DecodeIOBinding(
    Ort::Value batch_indicator, int32_t num_feature_frames) {
  int32_t *batch_indicator_ptr =
      batch_indicator.GetTensorMutableData<int32_t>();
  int32_t batch_size = batch_indicator_ptr[0];
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<int64_t> initial_tokens = model_->GetInitialTokens();
  if (model_->IsMultiLingual()) {
    if (!config_.language.empty()) {
      const auto &lang2id = model_->GetLang2ID();

      if (!lang2id.count(config_.language)) {
        SHERPA_ONNX_LOGE("Invalid language: %s", config_.language.c_str());
        exit(-1);
      }

      int32_t lang_id = lang2id.at(config_.language);

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    } else {
      int32_t lang_id = model_->DetectLanguageWithBinding()[0];

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    }

    if (config_.task != "transcribe") {
      // initial_tokens[2] is transcribe by default
      SHERPA_ONNX_LOGE(
          "Unsupported task: %s. Valid values are: transcribe, translate.",
          config_.task.c_str());
    }
  }

  initial_tokens.push_back(model_->NoTimeStampsToken());

  std::vector<int64_t> token_buffer(batch_size);
  std::array<int64_t, 2> token_shape{batch_size, static_cast<int64_t>(1)};
  std::vector<int32_t> predicted_tokens_one_step(batch_size);

  for (int32_t i = 0; i < initial_tokens.size(); i++) {
    for (int32_t b = 0; b < batch_size; b++) {
      token_buffer[b] = initial_tokens[i];
    }
    Ort::Value tokens = Ort::Value::CreateTensor(
        memory_info, token_buffer.data(), token_buffer.size(),
        token_shape.data(), token_shape.size());
    auto decoder_out = model_->ForwardDecoderWithBinding(std::move(tokens));
    const auto &logits = std::get<0>(decoder_out);
    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    if (i == initial_tokens.size() - 1) {
      int32_t vocab_size = logits_shape[2];
      const float *p_logits = logits.GetTensorData<float>();
      for (int32_t j = 0; j < batch_size; ++j) {
        predicted_tokens_one_step[j] = static_cast<int32_t>(
            std::distance(p_logits + (j * vocab_size),
                          std::max_element(p_logits + (j * vocab_size),
                                           p_logits + (j + 1) * vocab_size)));
      }
    }
  }

  std::vector<std::vector<int32_t>> predicted_tokens(batch_size);
  std::vector<bool> stop_signs(batch_size, false);

  // assume at most 6 tokens per second
  int32_t num_possible_tokens = num_feature_frames / 100 * 6;
  int32_t n_text_ctx = model_->TextCtx();
  num_possible_tokens = std::min<int32_t>(num_possible_tokens, n_text_ctx / 2);
  int32_t eot = model_->EOT();
  for (int32_t i = initial_tokens.size();
       i < initial_tokens.size() + num_possible_tokens; ++i) {
    bool all_eot = true;
    for (int32_t b = 0; b < batch_size; b++) {
      const auto &token = predicted_tokens_one_step[b];
      if (token != eot) {
        all_eot = false;
        if (!stop_signs[b]) {
          predicted_tokens[b].push_back(token);
        }
      } else {
        stop_signs[b] = true;
      }
      token_buffer[b] = token;
    }
    if (all_eot) {
      break;
    }
    Ort::Value tokens = Ort::Value::CreateTensor(
        memory_info, token_buffer.data(), token_buffer.size(),
        token_shape.data(), token_shape.size());
    auto decoder_out = model_->ForwardDecoderWithBinding(std::move(tokens));
    const auto &logits = std::get<0>(decoder_out);
    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    int32_t vocab_size = logits_shape[2];
    const float *p_logits = logits.GetTensorData<float>();
    for (int32_t j = 0; j < batch_size; ++j) {
      predicted_tokens_one_step[j] = static_cast<int32_t>(
          std::distance(p_logits + (j * vocab_size),
                        std::max_element(p_logits + (j * vocab_size),
                                         p_logits + (j + 1) * vocab_size)));
    }
  }
  std::vector<OfflineWhisperDecoderResult> ans(batch_size);

  const auto &id2lang = model_->GetID2Lang();
  std::string lang = "";
  if (id2lang.count(initial_tokens[1])) {
    lang = id2lang.at(initial_tokens[1]);
  }

  for (int32_t b = 0; b < batch_size; ++b) {
    ans[b].lang = lang;
    ans[b].tokens = std::move(predicted_tokens[b]);
  }
  return ans;
}

std::vector<OfflineWhisperDecoderResult>
OfflineWhisperGreedySearchDecoderOpt::DecodeOrig(Ort::Value cross_k,
                                                 Ort::Value cross_v,
                                                 int32_t num_feature_frames) {
  // cross_k, cross_v: (n_text_layer, N, n_audio_ctx, n_text_state)
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  auto shape = cross_k.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = shape[1];
  // For multilingual models, initial_tokens contains [sot, language, task]
  //   - language is English by default
  //   - task is transcribe by default
  //
  // For non-multilingual models, initial_tokens contains [sot]
  std::vector<int64_t> initial_tokens = model_->GetInitialTokens();

  if (model_->IsMultiLingual()) {
    if (!config_.language.empty()) {
      const auto &lang2id = model_->GetLang2ID();

      if (!lang2id.count(config_.language)) {
        SHERPA_ONNX_LOGE("Invalid language: %s", config_.language.c_str());
        exit(-1);
      }

      int32_t lang_id = lang2id.at(config_.language);

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    } else {
      int32_t lang_id = model_->DetectLanguage(cross_k, cross_v)[0];

      // 0: sot, 1: lang_id, 2: task, 3: no_timestamps
      initial_tokens[1] = lang_id;
    }

    if (config_.task == "translate") {
      initial_tokens[2] = model_->Translate();
    } else if (config_.task != "transcribe") {
      // initial_tokens[2] is transcribe by default
      SHERPA_ONNX_LOGE(
          "Unsupported task: %s. Valid values are: transcribe, translate.",
          config_.task.c_str());
    }
  }

  initial_tokens.push_back(model_->NoTimeStampsToken());

  std::vector<int64_t> token_buffer(batch_size);
  std::array<int64_t, 2> token_shape{batch_size, static_cast<int64_t>(1)};
  Ort::Value tokens = Ort::Value::CreateTensor(
      memory_info, token_buffer.data(), token_buffer.size(), token_shape.data(),
      token_shape.size());

  auto self_kv_cache = model_->GetInitialSelfKVCache(batch_size);
  // self_k_cache, self_v_cache: (n_text_layer, N, n_text_ctx, n_text_state)
  auto self_k_cache = std::move(self_kv_cache.first);
  auto self_v_cache = std::move(self_kv_cache.second);
  auto self_kv_shape = self_k_cache.GetTensorTypeAndShapeInfo().GetShape();
  int32_t num_layers = self_kv_shape[0];
  int32_t n_text_ctx = self_kv_shape[2];
  int32_t n_text_state = self_kv_shape[3];

  int64_t offset_value = 0;
  std::array<int64_t, 1> offset_shape{1};
  Ort::Value offset = Ort::Value::CreateTensor(
      memory_info, &offset_value, 1, offset_shape.data(), offset_shape.size());

  int32_t mask_value = 1;
  std::array<int64_t, 1> mask_shape{1};
  Ort::Value mask = Ort::Value::CreateTensor(
      memory_info, &mask_value, 1, mask_shape.data(), mask_shape.size());

  std::array<int64_t, 3> sel_shape{1, n_text_ctx, 1};
  Ort::Value sel = Ort::Value::CreateTensor<bool>(
      model_->Allocator(), sel_shape.data(), sel_shape.size());
  bool *sel_data = sel.GetTensorMutableData<bool>();
  std::fill(sel_data, sel_data + sel_shape[0] * sel_shape[1] * sel_shape[2],
            false);

  std::vector<int32_t> predicted_tokens_one_step(batch_size);

  for (int32_t i = 0; i < initial_tokens.size(); i++) {
    for (int32_t j = 0; j < batch_size; j++) {
      token_buffer[j] = initial_tokens[i];
    }

    sel_data[i] = true;
    auto decoder_out = model_->ForwardDecoder(
        std::move(View(&tokens)), std::move(View(&self_k_cache)),
        std::move(View(&self_v_cache)), std::move(View(&cross_k)),
        std::move(View(&cross_v)), std::move(View(&offset)),
        std::move(View(&mask)), std::move(View(&sel)));
    const auto &logits = std::get<0>(decoder_out);
    const auto &logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    const auto &k = std::get<1>(decoder_out);
    const auto &v = std::get<2>(decoder_out);
    float *p_k = std::get<1>(decoder_out).GetTensorMutableData<float>();
    float *p_v = std::get<2>(decoder_out).GetTensorMutableData<float>();
    const auto &kshape = k.GetTensorTypeAndShapeInfo().GetShape();
    const auto &vshape = v.GetTensorTypeAndShapeInfo().GetShape();

    if (i == initial_tokens.size() - 1) {
      int32_t vocab_size = logits_shape[2];
      const float *p_logits = logits.GetTensorData<float>();
      for (int32_t j = 0; j < batch_size; ++j) {
        predicted_tokens_one_step[j] = static_cast<int32_t>(
            std::distance(p_logits + (j * vocab_size),
                          std::max_element(p_logits + (j * vocab_size),
                                           p_logits + (j + 1) * vocab_size)));
      }
    }
    // self_k_cache, self_v_cache: (n_text_layer, N, n_text_ctx, n_text_state)
    // k, v: (n_text_layer, N, 1, n_text_state)
    // Copy the key and value tensors to the self attention cache, at
    // (n_text_layer, N, i, n_text_state)
    for (int32_t l = 0; l < num_layers; ++l) {
      for (int32_t b = 0; b < batch_size; ++b) {
        float *k_src = p_k + (l * batch_size + b) * n_text_state;
        float *k_dst = self_k_cache.GetTensorMutableData<float>() +
                       ((l * batch_size + b) * n_text_ctx + i) * n_text_state;
        float *v_src = p_v + (l * batch_size + b) * n_text_state;
        float *v_dst = self_v_cache.GetTensorMutableData<float>() +
                       ((l * batch_size + b) * n_text_ctx + i) * n_text_state;
        std::copy(k_src, k_src + n_text_state, k_dst);
        std::copy(v_src, v_src + n_text_state, v_dst);
      }
    }
    offset_value += 1;
    mask_value += 1;
    sel_data[i] = false;
  }

  std::vector<std::vector<int32_t>> predicted_tokens(batch_size);
  std::vector<bool> stop_signs(batch_size, false);

  // assume at most 6 tokens per second
  int32_t num_possible_tokens = num_feature_frames / 100 * 6;
  num_possible_tokens = std::min<int32_t>(num_possible_tokens, n_text_ctx / 2);
  int32_t eot = model_->EOT();
  for (int32_t i = initial_tokens.size();
       i < initial_tokens.size() + num_possible_tokens; ++i) {
    bool all_eot = true;
    for (int32_t b = 0; b < batch_size; b++) {
      const auto &token = predicted_tokens_one_step[b];
      if (token != eot) {
        all_eot = false;
        if (!stop_signs[b]) {
          predicted_tokens[b].push_back(token);
        }
      } else {
        stop_signs[b] = true;
      }
      token_buffer[b] = token;
    }
    if (all_eot) {
      break;
    }

    sel_data[i] = true;
    auto decoder_out = model_->ForwardDecoder(
        std::move(View(&tokens)), std::move(View(&self_k_cache)),
        std::move(View(&self_v_cache)), std::move(View(&cross_k)),
        std::move(View(&cross_v)), std::move(View(&offset)),
        std::move(View(&mask)), std::move(View(&sel)));
    const auto &logits = std::get<0>(decoder_out);
    float *k = std::get<1>(decoder_out).GetTensorMutableData<float>();
    float *v = std::get<2>(decoder_out).GetTensorMutableData<float>();

    auto logits_shape = logits.GetTensorTypeAndShapeInfo().GetShape();
    int32_t vocab_size = logits_shape[2];
    const float *p_logits = logits.GetTensorData<float>();
    for (int32_t j = 0; j < batch_size; ++j) {
      predicted_tokens_one_step[j] = static_cast<int32_t>(
          std::distance(p_logits + (j * vocab_size),
                        std::max_element(p_logits + (j * vocab_size),
                                         p_logits + (j + 1) * vocab_size)));
    }

    for (int32_t l = 0; l < num_layers; ++l) {
      for (int32_t b = 0; b < batch_size; ++b) {
        float *k_src = k + (l * batch_size + b) * n_text_state;
        float *k_dst = self_k_cache.GetTensorMutableData<float>() +
                       ((l * batch_size + b) * n_text_ctx + i) * n_text_state;
        float *v_src = v + (l * batch_size + b) * n_text_state;
        float *v_dst = self_v_cache.GetTensorMutableData<float>() +
                       ((l * batch_size + b) * n_text_ctx + i) * n_text_state;
        std::copy(k_src, k_src + n_text_state, k_dst);
        std::copy(v_src, v_src + n_text_state, v_dst);
      }
    }
    offset_value += 1;
    mask_value += 1;
    sel_data[i] = false;
  }

  std::vector<OfflineWhisperDecoderResult> ans(batch_size);

  const auto &id2lang = model_->GetID2Lang();
  std::string lang = "";
  if (id2lang.count(initial_tokens[1])) {
    lang = id2lang.at(initial_tokens[1]);
  }

  for (int32_t b = 0; b < batch_size; ++b) {
    ans[b].lang = lang;
    ans[b].tokens = std::move(predicted_tokens[b]);
  }
  return ans;
}

}  // namespace sherpa_onnx
