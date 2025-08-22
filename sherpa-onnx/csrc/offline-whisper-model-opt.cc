// sherpa-onnx/csrc/offline-whisper-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-model-opt.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
#include "sherpa-onnx/csrc/dml-utils.h"  // NOLINT
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineWhisperModelOpt::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    int32_t device_id = 0;
    if (config.io_binding) {
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
      dml_mem_manager_ = std::make_unique<DmlMemManager>();
      dml_mem_manager_->Initialize(device_id);
      sess_opts_ = dml_mem_manager_->CreateSessionOptions();
#endif
    }
    {
      // auto buf = ReadFile(config.whisper.encoder);
      // InitEncoder(buf.data(), buf.size());
      InitEncoder(config.whisper.encoder);
    }

    {
      // auto buf = ReadFile(config.whisper.decoder);
      // InitDecoder(buf.data(), buf.size());
      InitDecoder(config.whisper.decoder);
    }
    if (config_.io_binding) {
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
      int32_t max_num_frames = 3000;
      int32_t n_audio_ctx = max_num_frames / 2;
      int32_t max_batch_size = config_.whisper.max_batch_size;
      Ort::MemoryInfo dml_mem("DML", OrtDeviceAllocator, device_id,
                              OrtMemTypeDefault);
      std::array<int64_t, 3> feature_shape = {max_batch_size, n_mels_,
                                              max_num_frames};
      feature_mem_ = dml_mem_manager_->AllocateDmlMem(
          max_batch_size * max_num_frames * n_mels_ * sizeof(float));
      Ort::Value feature_tensor = Ort::Value::CreateTensor(
          dml_mem, static_cast<float *>(feature_mem_.data_),
          max_batch_size * n_mels_ * max_num_frames, feature_shape.data(),
          feature_shape.size());

      encoder_input_tensors_.push_back(std::move(feature_tensor));

      std::array<int64_t, 4> cross_kv_shape = {n_text_layer_, max_batch_size,
                                               n_audio_ctx, n_text_state_};
      cross_k_mem_ = dml_mem_manager_->AllocateDmlMem(
          n_text_layer_ * max_batch_size * n_audio_ctx * n_text_state_ *
          sizeof(float));
      cross_v_mem_ = dml_mem_manager_->AllocateDmlMem(
          n_text_layer_ * max_batch_size * n_audio_ctx * n_text_state_ *
          sizeof(float));
      Ort::Value cross_k_tensor = Ort::Value::CreateTensor(
          dml_mem, static_cast<float *>(cross_k_mem_.data_),
          n_text_layer_ * max_batch_size * n_audio_ctx * n_text_state_,
          cross_kv_shape.data(), cross_kv_shape.size());
      Ort::Value cross_v_tensor = Ort::Value::CreateTensor(
          dml_mem, static_cast<float *>(cross_v_mem_.data_),
          n_text_layer_ * max_batch_size * n_audio_ctx * n_text_state_,
          cross_kv_shape.data(), cross_kv_shape.size());

      cross_kv_tensors_.push_back(std::move(cross_k_tensor));
      cross_kv_tensors_.push_back(std::move(cross_v_tensor));

      std::array<int64_t, 2> tokens_shape = {max_batch_size, 1};
      tokens_mem_ =
          dml_mem_manager_->AllocateDmlMem(max_batch_size * sizeof(int64_t));
      Ort::Value tokens_tensor = Ort::Value::CreateTensor<int64_t>(
          dml_mem, static_cast<int64_t *>(tokens_mem_.data_), max_batch_size,
          tokens_shape.data(), tokens_shape.size());

      std::array<int64_t, 4> self_kv_shape = {n_text_layer_, max_batch_size,
                                              n_text_ctx_, n_text_state_};
      self_k_mem_ = dml_mem_manager_->AllocateDmlMem(
          n_text_layer_ * max_batch_size * n_text_ctx_ * n_text_state_ *
          sizeof(float));
      self_v_mem_ = dml_mem_manager_->AllocateDmlMem(
          n_text_layer_ * max_batch_size * n_text_ctx_ * n_text_state_ *
          sizeof(float));
      Ort::Value self_k_tensor = Ort::Value::CreateTensor<float>(
          dml_mem, static_cast<float *>(self_k_mem_.data_),
          n_text_layer_ * max_batch_size * n_text_ctx_ * n_text_state_,
          self_kv_shape.data(), self_kv_shape.size());
      Ort::Value self_v_tensor = Ort::Value::CreateTensor<float>(
          dml_mem, static_cast<float *>(self_v_mem_.data_),
          n_text_layer_ * max_batch_size * n_text_ctx_ * n_text_state_,
          self_kv_shape.data(), self_kv_shape.size());

      decoder_input_tensors_partial_.push_back(std::move(tokens_tensor));
      decoder_input_tensors_partial_.push_back(std::move(self_k_tensor));
      decoder_input_tensors_partial_.push_back(std::move(self_v_tensor));

      std::array<int64_t, 4> logits_shape{max_batch_size, 1, n_vocab_};
      logits_mem_ = dml_mem_manager_->AllocateDmlMem(max_batch_size * n_vocab_ *
                                                     sizeof(float));
      Ort::Value logits_tensor = Ort::Value::CreateTensor<float>(
          dml_mem, static_cast<float *>(logits_mem_.data_),
          max_batch_size * n_vocab_, logits_shape.data(), logits_shape.size());

      std::array<int64_t, 4> out_self_kv_shape{n_text_layer_, max_batch_size, 1,
                                               n_text_state_};
      out_self_k_mem_ = dml_mem_manager_->AllocateDmlMem(
          n_text_layer_ * max_batch_size * n_text_state_ * sizeof(float));
      out_self_v_mem_ = dml_mem_manager_->AllocateDmlMem(
          n_text_layer_ * max_batch_size * n_text_state_ * sizeof(float));
      Ort::Value out_self_k_tensor = Ort::Value::CreateTensor<float>(
          dml_mem, static_cast<float *>(out_self_k_mem_.data_),
          n_text_layer_ * max_batch_size * n_text_state_,
          out_self_kv_shape.data(), out_self_kv_shape.size());
      Ort::Value out_self_v_tensor = Ort::Value::CreateTensor<float>(
          dml_mem, static_cast<float *>(out_self_v_mem_.data_),
          n_text_layer_ * max_batch_size * n_text_state_,
          out_self_kv_shape.data(), out_self_kv_shape.size());
      decoder_output_tensors_.push_back(std::move(logits_tensor));
      decoder_output_tensors_.push_back(std::move(out_self_k_tensor));
      decoder_output_tensors_.push_back(std::move(out_self_v_tensor));

      // NOTE(LX):
      //  offset, mask, sel will be created dynamicly.
      //  So no need to generate Ort::Value for now.
      std::vector<int64_t> offset_buffer;
      offset_buffer.reserve(n_text_ctx_);
      std::vector<int32_t> attention_mask_buffer;
      attention_mask_buffer.reserve(n_text_ctx_);
      std::vector<bool> sel_buffer;
      sel_buffer.reserve(n_text_ctx_ * n_text_ctx_);
      for (int i = 0; i < n_text_ctx_; ++i) {
        offset_buffer.push_back(i);
        attention_mask_buffer.push_back(i + 1);
        for (int j = 0; j < n_text_ctx_; ++j) {
          sel_buffer.push_back(i == j);
        }
      }
      offset_mem_ =
          dml_mem_manager_->AllocateDmlMem(n_text_ctx_ * sizeof(int64_t));
      dml_mem_manager_->CopyToGPU(offset_buffer.data(), &offset_mem_,
                                  n_text_ctx_ * sizeof(int64_t));

      attention_mask_mem_ =
          dml_mem_manager_->AllocateDmlMem(n_text_ctx_ * sizeof(int32_t));
      dml_mem_manager_->CopyToGPU(attention_mask_buffer.data(),
                                  &attention_mask_mem_,
                                  n_text_ctx_ * sizeof(int32_t));
      sel_mem_ = dml_mem_manager_->AllocateDmlMem(n_text_ctx_ * n_text_ctx_ *
                                                  sizeof(bool));
      dml_mem_manager_->CopyToGPU(sel_buffer.data(), &sel_mem_,
                                  n_text_ctx_ * n_text_ctx_ * sizeof(bool));
      // Wait to complete all
      dml_mem_manager_->WaitForGPU();
      encoder_io_binding_ = Ort::IoBinding(*encoder_sess_);
      encoder_io_binding_.BindInput(encoder_input_names_ptr_[0],
                                    encoder_input_tensors_[0]);
      encoder_io_binding_.BindOutput(encoder_output_names_ptr_[0],
                                     cross_kv_tensors_[0]);
      encoder_io_binding_.BindOutput(encoder_output_names_ptr_[1],
                                     cross_kv_tensors_[1]);
#endif
    }
  }

  explicit Impl(const SpokenLanguageIdentificationConfig &config)
      : lid_config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.whisper.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.whisper.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.whisper.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.whisper.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const SpokenLanguageIdentificationConfig &config)
      : lid_config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.whisper.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.whisper.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  std::pair<Ort::Value, Ort::Value> ForwardEncoder(Ort::Value features) {
    auto encoder_out = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), &features, 1,
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());

    return {std::move(encoder_out[0]), std::move(encoder_out[1])};
  }

  void ForwardEncoderWithBinding(Ort::Value features) {
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
    if (config_.io_binding) {
      dml_mem_manager_->CopyToGPU(features.GetTensorData<float>(),
                                  &encoder_input_mem_,
                                  features.GetTensorSizeInBytes());
      dml_mem_manager_->WaitForGPU();
      dml_mem_manager_->FlushCommandLists();
      Ort::RunOptions ro;
      encoder_sess_->Run(ro, encoder_io_binding_);
      encoder_io_binding_.SynchronizeOutputs();
      dml_mem_manager_->WaitForGPU();
      step_ = 0;
    } else {
#endif
      printf(
          "Error: ForwardEncoderWithBinding must run with DML and IO-Binding");
      throw std::runtime_error(
          "ForwardEncoderWithBinding must run with DML and IO-Binding");
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
    }
#endif
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value> ForwardDecoder(
      Ort::Value tokens, Ort::Value n_layer_self_k_cache,
      Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
      Ort::Value n_layer_cross_v, Ort::Value offset, Ort::Value attention_mask,
      Ort::Value sel) {
    std::array<Ort::Value, 8> decoder_input = {std::move(tokens),
                                               std::move(n_layer_self_k_cache),
                                               std::move(n_layer_self_v_cache),
                                               std::move(n_layer_cross_k),
                                               std::move(n_layer_cross_v),
                                               std::move(offset),
                                               std::move(attention_mask),
                                               std::move(sel)};

    auto decoder_out = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), decoder_input.data(),
        decoder_input.size(), decoder_output_names_ptr_.data(),
        decoder_output_names_ptr_.size());

    return std::tuple<Ort::Value, Ort::Value, Ort::Value>{
        std::move(decoder_out[0]), std::move(decoder_out[1]),
        std::move(decoder_out[2])};
  }

  std::tuple<Ort::Value> ForwardDecoderWithBinding(Ort::Value tokens) {
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
    if (config_.io_binding) {
      auto tokens_shape = tokens.GetTensorTypeAndShapeInfo().GetShape();
      int batch_size = tokens_shape[0];
      if (batch_size != config_.whisper.max_batch_size) {
        throw std::runtime_error("Invalid batch size");
      }

      dml_mem_manager->CopyToGPU(tokens.GetTensorData<int64_t>(), &tokens_mem_,
                                 tokens.GetTensorSizeInBytes());
      dml_mem_manager_->WaitForGPU();
      dml_mem_manager_->FlushCommandLists();

      Ort::IoBinding decoder_io_binding(*decoder_sess_);
      Ort::MemoryInfo dml_mem("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);

      // token and self kv cache
      decoder_io_binding.BindInput(decoder_input_names_ptr_[0],
                                   decoder_input_tensors_partial_[0]);
      decoder_io_binding.BindOutput(decoder_input_names_ptr_[1],
                                    decoder_input_tensors_partial_[1]);
      decoder_io_binding.BindOutput(decoder_input_names_ptr_[2],
                                    decoder_input_tensors_partial_[2]);
      // cross kv cache
      decoder_io_binding.BindOutput(decoder_input_names_ptr_[3],
                                    cross_kv_tensors_[0]);
      decoder_io_binding.BindOutput(decoder_input_names_ptr_[4],
                                    cross_kv_tensors_[1]);

      // offset, mask, sel
      std::array<int64_t, 1> offset_shape{1};
      Ort::Value offset_tensor = Ort::Value::CreateTensor<int64_t>(
          dml_mem, static_cast<int64_t *>(offset_mem_.data_) + step_, 1,
          offset_shape.data(), offset_shape.size());

      std::array<int64_t, 1> mask_shape{1};
      Ort::Value mask_tensor = Ort::Value::CreateTensor<int32_t>(
          dml_mem, static_cast<int32_t *>(attention_mask_mem_.data_) + step_, 1,
          mask_shape.data(), mask_shape.size());

      std::array<int64_t, 3> sel_shape{1, n_text_ctx_, 1};
      Ort::Value sel_tensor = Ort::Value::CreateTensor<bool>(
          dml_mem, static_cast<bool *>(sel_mem_.data_) + step_ * n_text_ctx_,
          n_text_ctx_, sel_shape.data(), sel_shape.size());

      decoder_io_binding_.BindInput(decoder_input_names_ptr_[5], offset_tensor);
      decoder_io_binding_.BindInput(decoder_input_names_ptr_[6], mask_tensor);
      decoder_io_binding_.BindInput(decoder_input_names_ptr_[7], sel_tensor);
      // logits, out_self_k, out_self_v
      decoder_io_binding_.BindOutput(decoder_input_names_ptr_[8],
                                     logits_tensor);
      decoder_io_binding_.BindOutput(decoder_input_names_ptr_[9],
                                     out_self_k_tensor);
      decoder_io_binding_.BindOutput(decoder_input_names_ptr_[10],
                                     out_self_v_tensor);
      decoder_io_binding_.SynchronizeInputs();
      Ort::RunOptions ro;
      decoder_sess_->Run(ro, decoder_io_binding_);
      decoder_io_binding_.SynchronizeOutputs();

      // TODO(LX): copy outputs to host
      // TODO(LX): copy out_self_kv_cache to self_kv_cache

      std::array<int64_t, 2> logits_shape{batch_size, n_vocab_};
      Ort::Value logits = Ort::Value::CreateTensor<bool>(
          model_->Allocator(), logits_shape.data(), logits_shape.size());
      dml_mem_manager_->CopyFromGPU(logits_mem_.data_,
                                    logits.GetTensorMutableData<bool>(),
                                    logits.GetTensorSizeInBytes());
      dml_mem_manager_->WaitForGPU();
      dml_mem_manager_->FlushCommandLists();
      for (int32_t l = 0; l < n_text_layer_; ++l) {
        for (int32_t b = 0; b < batch_size; ++b) {
          DmlMem k_src = out_self_k_mem_;
          DmlMem v_src = out_self_v_mem_;
          DmlMem k_dst = self_k_mem_;
          DmlMem v_dst = self_v_mem_;
          k_src.data_ = static_cast<float *>(out_self_k_mem_.data_) +
                        (l * batch_size + b) * n_text_state_;
          k_dst.data_ =
              static_cast<float *>(self_k_mem_.data_) +
              ((l * batch_size + b) * n_text_ctx_ + i) * n_text_state_;
          v_src.data_ = static_cast<float *>(out_self_v_mem_.data_) +
                        (l * batch_size + b) * n_text_state_;
          v_dst.data_ =
              static_cast<float *>(self_v_mem_.data_) +
              ((l * batch_size + b) * n_text_ctx_ + i) * n_text_state_;
          dml_mem_manager_->CopyFromGPUToGPU(&k_src, &k_dst,
                                             n_text_state_ * sizeof(float));
          dml_mem_manager_->CopyFromGPUToGPU(&v_src, &v_dst,
                                             n_text_state_ * sizeof(float));
        }
      }
      dml_mem_manager_->WaitForGPU();
      dml_mem_manager_->FlushCommandLists();
      step_ += 1;
      return std::tuple<Ort::Value>{std::move(logits)};
    } else {
#endif
      printf(
          "Error: ForwardDecoderWithBinding must run with DML and IO-Binding");
      throw std::runtime_error(
          "ForwardDecoderWithBinding must run with DML and IO-Binding");
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
    }
#endif
  }

  std::vector<int32_t> DetectLanguage(Ort::Value &cross_k,    // NOLINT
                                      Ort::Value &cross_v) {  // NOLINT
    auto kv_shape = cross_k.GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = kv_shape[1];

    std::vector<int64_t> token_val(batch_size, SOT());
    std::array<int64_t, 2> token_shape{batch_size, 1};

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value tokens = Ort::Value::CreateTensor(
        memory_info, token_val.data(), token_val.size(), token_shape.data(),
        token_shape.size());

    auto self_kv_cache = GetInitialSelfKVCache();

    std::array<int64_t, 1> offset_shape{1};
    Ort::Value offset = Ort::Value::CreateTensor<int64_t>(
        Allocator(), offset_shape.data(), offset_shape.size());
    *(offset.GetTensorMutableData<int64_t>()) = 0;

    std::array<int64_t, 1> attention_mask_shape{1};
    Ort::Value attention_mask = Ort::Value::CreateTensor<int32_t>(
        Allocator(), attention_mask_shape.data(), attention_mask_shape.size());
    *(attention_mask.GetTensorMutableData<int32_t>()) = 1;

    std::array<int64_t, 3> sel_shape{1, n_text_ctx_, 1};
    Ort::Value sel = Ort::Value::CreateTensor<bool>(
        Allocator(), sel_shape.data(), sel_shape.size());
    *(sel.GetTensorMutableData<bool>()) = true;

    auto decoder_out = ForwardDecoder(
        std::move(tokens), std::move(self_kv_cache.first),
        std::move(self_kv_cache.second), std::move(View(&cross_k)),
        std::move(View(&cross_v)), std::move(offset), std::move(attention_mask),
        std::move(sel));

    const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();
    const auto &p_logits_shape =
        std::get<0>(decoder_out).GetTensorTypeAndShapeInfo().GetShape();
    const int32_t vocab_size = p_logits_shape[2];
    const auto &all_language_ids = GetAllLanguageIDs();

    std::vector<int32_t> result(batch_size, all_language_ids[0]);

    for (int32_t i = 0; i < batch_size; ++i) {
      int32_t lang_id = all_language_ids[0];
      const float *p_logits_batch = p_logits + i * vocab_size;
      float this_logit = p_logits_batch[lang_id];
      for (int32_t j = 1; j != all_language_ids.size(); ++j) {
        int32_t id = all_language_ids[j];
        float p = p_logits_batch[id];

        if (p > this_logit) {
          this_logit = p;
          lang_id = id;
        }
      }
      result[i] = lang_id;
    }

    return result;
  }

  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache(
      const int32_t batch_size = 1) {
    std::array<int64_t, 4> shape{n_text_layer_, batch_size, n_text_ctx_,
                                 n_text_state_};

    Ort::Value n_layer_self_k_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    Ort::Value n_layer_self_v_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    auto n = shape[0] * shape[1] * shape[2] * shape[3];

    float *p_k = n_layer_self_k_cache.GetTensorMutableData<float>();
    float *p_v = n_layer_self_v_cache.GetTensorMutableData<float>();

    memset(p_k, 0, sizeof(float) * n);
    memset(p_v, 0, sizeof(float) * n);

    return {std::move(n_layer_self_k_cache), std::move(n_layer_self_v_cache)};
  }

  OrtAllocator *Allocator() { return allocator_; }

  const std::vector<int64_t> &GetInitialTokens() const { return sot_sequence_; }

  const std::vector<int32_t> &GetAllLanguageIDs() const {
    return all_language_tokens_;
  }

  const std::unordered_map<std::string, int32_t> &GetLang2ID() const {
    return lang2id_;
  }

  const std::unordered_map<int32_t, std::string> &GetID2Lang() const {
    return id2lang_;
  }

  int32_t NoTimeStampsToken() const { return no_timestamps_; }

  int32_t EOT() const { return eot_; }

  int32_t SOT() const { return sot_; }

  int32_t TextCtx() const { return n_text_ctx_; }

  int32_t VocabSize() const { return n_vocab_; }

  int32_t FeatureDim() const { return n_mels_; }

  int32_t Translate() const { return translate_; }

  bool IsMultiLingual() const { return is_multilingual_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---encoder---\n";
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(n_mels_, "n_mels");
    SHERPA_ONNX_READ_META_DATA(n_text_layer_, "n_text_layer");
    SHERPA_ONNX_READ_META_DATA(n_text_ctx_, "n_text_ctx");
    SHERPA_ONNX_READ_META_DATA(n_text_state_, "n_text_state");
    SHERPA_ONNX_READ_META_DATA(n_vocab_, "n_vocab");
    SHERPA_ONNX_READ_META_DATA(sot_, "sot");
    SHERPA_ONNX_READ_META_DATA(eot_, "eot");
    SHERPA_ONNX_READ_META_DATA(blank_, "blank_id");
    SHERPA_ONNX_READ_META_DATA(translate_, "translate");
    SHERPA_ONNX_READ_META_DATA(transcribe_, "transcribe");
    SHERPA_ONNX_READ_META_DATA(is_multilingual_, "is_multilingual");
    SHERPA_ONNX_READ_META_DATA(no_timestamps_, "no_timestamps");
    SHERPA_ONNX_READ_META_DATA(no_speech_, "no_speech");
    SHERPA_ONNX_READ_META_DATA_VEC(sot_sequence_, "sot_sequence");

    if (is_multilingual_) {
      SHERPA_ONNX_READ_META_DATA_VEC(all_language_tokens_,
                                     "all_language_tokens");
      SHERPA_ONNX_READ_META_DATA_VEC_STRING(all_language_codes_,
                                            "all_language_codes");
      if (all_language_tokens_.size() != all_language_codes_.size()) {
        SHERPA_ONNX_LOGE("# lang_id: %d != # lang_code: %d",
                         static_cast<int32_t>(all_language_tokens_.size()),
                         static_cast<int32_t>(all_language_codes_.size()));
        exit(-1);
      }

      for (int32_t i = 0;
           i != static_cast<int32_t>(all_language_tokens_.size()); ++i) {
        lang2id_[all_language_codes_[i]] = all_language_tokens_[i];
        id2lang_[all_language_tokens_[i]] = all_language_codes_[i];
      }
    }
  }

  void InitEncoder(const std::string &model_path) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, std::filesystem::path(model_path).c_str(), sess_opts_);

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---encoder---\n";
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(n_mels_, "n_mels");
    SHERPA_ONNX_READ_META_DATA(n_text_layer_, "n_text_layer");
    SHERPA_ONNX_READ_META_DATA(n_text_ctx_, "n_text_ctx");
    SHERPA_ONNX_READ_META_DATA(n_text_state_, "n_text_state");
    SHERPA_ONNX_READ_META_DATA(n_vocab_, "n_vocab");
    SHERPA_ONNX_READ_META_DATA(sot_, "sot");
    SHERPA_ONNX_READ_META_DATA(eot_, "eot");
    SHERPA_ONNX_READ_META_DATA(blank_, "blank_id");
    SHERPA_ONNX_READ_META_DATA(translate_, "translate");
    SHERPA_ONNX_READ_META_DATA(transcribe_, "transcribe");
    SHERPA_ONNX_READ_META_DATA(is_multilingual_, "is_multilingual");
    SHERPA_ONNX_READ_META_DATA(no_timestamps_, "no_timestamps");
    SHERPA_ONNX_READ_META_DATA(no_speech_, "no_speech");
    SHERPA_ONNX_READ_META_DATA_VEC(sot_sequence_, "sot_sequence");

    if (is_multilingual_) {
      SHERPA_ONNX_READ_META_DATA_VEC(all_language_tokens_,
                                     "all_language_tokens");
      SHERPA_ONNX_READ_META_DATA_VEC_STRING(all_language_codes_,
                                            "all_language_codes");
      if (all_language_tokens_.size() != all_language_codes_.size()) {
        SHERPA_ONNX_LOGE("# lang_id: %d != # lang_code: %d",
                         static_cast<int32_t>(all_language_tokens_.size()),
                         static_cast<int32_t>(all_language_codes_.size()));
        exit(-1);
      }

      for (int32_t i = 0;
           i != static_cast<int32_t>(all_language_tokens_.size()); ++i) {
        lang2id_[all_language_codes_[i]] = all_language_tokens_[i];
        id2lang_[all_language_tokens_[i]] = all_language_codes_[i];
      }
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);
  }

  void InitDecoder(const std::string &model_path) {
    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, std::filesystem::path(model_path).c_str(), sess_opts_);

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);
  }

 private:
#if defined(_WIN32) && SHERPA_ONNX_ENABLE_DIRECTML == 1
  std::unique_ptr<DmlMemManager> dml_mem_manager_ = nullptr;
  // encoder input
  DmlMem feature_mem_;
  // encoder output
  DmlMem cross_k_mem_;
  DmlMem cross_v_mem_;
  // decoder output + two caches above
  DmlMem tokens_mem_;
  DmlMem self_k_mem_;
  DmlMem self_v_mem_;
  DmlMem offset_mem_;
  DmlMem attention_mask_mem_;
  DmlMem sel_mem_;
  // decoder output
  DmlMem logits_mem_;
  DmlMem out_self_k_mem_;
  DmlMem out_self_v_mem_;
  std::vector<Ort::Value> encoder_input_tensors_;
  std::vector<Ort::Value> cross_kv_tensors_;
  std::vector<Ort::Value> decoder_input_tensors_partial_;
  std::vector<Ort::Value> decoder_output_tensors_;
  Ort::IoBinding encoder_io_binding_;
  int32_t step_ = 0;
#endif
  OfflineModelConfig config_;
  SpokenLanguageIdentificationConfig lid_config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<int32_t> all_language_tokens_;
  std::vector<std::string> all_language_codes_;
  std::unordered_map<std::string, int32_t> lang2id_;
  std::unordered_map<int32_t, std::string> id2lang_;

  // model meta data
  int32_t n_mels_ = 80;
  int32_t n_text_layer_ = 0;
  int32_t n_text_ctx_ = 0;
  int32_t n_text_state_ = 0;
  int32_t n_vocab_ = 0;
  int32_t sot_ = 0;
  int32_t eot_ = 0;
  int32_t blank_ = 0;
  int32_t translate_ = 0;
  int32_t transcribe_ = 0;
  int32_t no_timestamps_ = 0;
  int32_t no_speech_ = 0;
  int32_t is_multilingual_ = 0;
  std::vector<int64_t> sot_sequence_;
};

OfflineWhisperModelOpt::OfflineWhisperModelOpt(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineWhisperModelOpt::OfflineWhisperModelOpt(
    const SpokenLanguageIdentificationConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModelOpt::OfflineWhisperModelOpt(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

template <typename Manager>
OfflineWhisperModelOpt::OfflineWhisperModelOpt(
    Manager *mgr, const SpokenLanguageIdentificationConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperModelOpt::~OfflineWhisperModelOpt() = default;

std::pair<Ort::Value, Ort::Value> OfflineWhisperModelOpt::ForwardEncoder(
    Ort::Value features) const {
  return impl_->ForwardEncoder(std::move(features));
}

std::tuple<Ort::Value, Ort::Value, Ort::Value>
OfflineWhisperModelOpt::ForwardDecoder(
    Ort::Value tokens, Ort::Value n_layer_self_k_cache,
    Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
    Ort::Value n_layer_cross_v, Ort::Value offset, Ort::Value attention_mask,
    Ort::Value sel) const {
  return impl_->ForwardDecoder(
      std::move(tokens), std::move(n_layer_self_k_cache),
      std::move(n_layer_self_v_cache), std::move(n_layer_cross_k),
      std::move(n_layer_cross_v), std::move(offset), std::move(attention_mask),
      std::move(sel));
}

std::vector<int32_t> OfflineWhisperModelOpt::DetectLanguage(
    Ort::Value &cross_k,    // NOLINT
    Ort::Value &cross_v) {  // NOLINT
  return impl_->DetectLanguage(cross_k, cross_v);
}

std::pair<Ort::Value, Ort::Value> OfflineWhisperModelOpt::GetInitialSelfKVCache(
    const int32_t batch_size) const {
  return impl_->GetInitialSelfKVCache(batch_size);
}

OrtAllocator *OfflineWhisperModelOpt::Allocator() const {
  return impl_->Allocator();
}

const std::vector<int64_t> &OfflineWhisperModelOpt::GetInitialTokens() const {
  return impl_->GetInitialTokens();
}

const std::vector<int32_t> &OfflineWhisperModelOpt::GetAllLanguageIDs() const {
  return impl_->GetAllLanguageIDs();
}

const std::unordered_map<std::string, int32_t> &
OfflineWhisperModelOpt::GetLang2ID() const {
  return impl_->GetLang2ID();
}

const std::unordered_map<int32_t, std::string> &
OfflineWhisperModelOpt::GetID2Lang() const {
  return impl_->GetID2Lang();
}

int32_t OfflineWhisperModelOpt::NoTimeStampsToken() const {
  return impl_->NoTimeStampsToken();
}

int32_t OfflineWhisperModelOpt::EOT() const { return impl_->EOT(); }

int32_t OfflineWhisperModelOpt::SOT() const { return impl_->SOT(); }

int32_t OfflineWhisperModelOpt::TextCtx() const { return impl_->TextCtx(); }

int32_t OfflineWhisperModelOpt::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineWhisperModelOpt::FeatureDim() const {
  return impl_->FeatureDim();
}

int32_t OfflineWhisperModelOpt::Translate() const { return impl_->Translate(); }

bool OfflineWhisperModelOpt::IsMultiLingual() const {
  return impl_->IsMultiLingual();
}

void OfflineWhisperModelOpt::NormalizeFeatures(float *features,
                                               int32_t num_frames,
                                               int32_t feat_dim) {
  // log_spec = torch.clamp(features, min=1e-10).log10()
  // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  // mel = (log_spec + 4.0) / 4.0

  int32_t n = num_frames * feat_dim;
  float max_v = -1e20;
  for (int32_t i = 0; i != n; ++i) {
    float f = features[i];

    f = std::max<float>(f, 1e-10);
    f = std::log10(f);

    max_v = std::max(f, max_v);

    features[i] = f;
  }

  max_v -= 8;

  for (int32_t i = 0; i != n; ++i) {
    float f = features[i];
    f = std::max(f, max_v);

    f = (f + 4) / 4;

    features[i] = f;
  }
}

#if __ANDROID_API__ >= 9
template OfflineWhisperModelOpt::OfflineWhisperModelOpt(
    AAssetManager *mgr, const OfflineModelConfig &config);

template OfflineWhisperModelOpt::OfflineWhisperModelOpt(
    AAssetManager *mgr, const SpokenLanguageIdentificationConfig &config);
#endif

#if __OHOS__
template OfflineWhisperModelOpt::OfflineWhisperModelOpt(
    NativeResourceManager *mgr, const OfflineModelConfig &config);

template OfflineWhisperModelOpt::OfflineWhisperModelOpt(
    NativeResourceManager *mgr,
    const SpokenLanguageIdentificationConfig &config);
#endif

}  // namespace sherpa_onnx
