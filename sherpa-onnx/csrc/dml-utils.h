#ifndef SHERPA_ONNX_DML_UTILS_H
#define SHERPA_ONNX_DML_UTILS_H
#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <iostream>

#include "dml_provider_factory.h"  // NOLINT
#include "onnxruntime_cxx_api.h"
namespace sherpa_onnx {
using Microsoft::WRL::ComPtr;

class DmlMem {
 public:
  DmlMem() = default;
  DmlMem(ComPtr<ID3D12Resource> d2c_res, void *data)
      : d2d_res_(d2c_res), data_(data) {}

 public:
  ComPtr<ID3D12Resource> d2d_res_;
  void *data_;
};

class DmlMemManager {
 public:
  DmlMemManager() = default;

  ~DmlMemManager() {}

  void Initialize(int adapter_index = 0) {
    // #ifdef _DEBUG
    // ComPtr<ID3D12Debug> debugController;
    // if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
    //   debugController->EnableDebugLayer();
    //   printf("D3D12 debug layer enabled\n");
    // }
    // #endif
    ComPtr<IDXGIFactory6> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
      printf("Failed to create DXGI factory: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create DXGI factory");
    }

    ComPtr<IDXGIAdapter1> adapter;
    hr = factory->EnumAdapters1(adapter_index, &adapter);
    if (FAILED(hr)) {
      printf("Failed to enumerate adapter %d: 0x%08X\n", adapter_index, hr);
      throw std::runtime_error("Failed to enumerate adapter");
    }

    // Print adapter info
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    wprintf(L"Using adapter: %s\n", desc.Description);

    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                           IID_PPV_ARGS(&d3d_device_));
    if (FAILED(hr)) {
      printf("Failed to create D3D12 device: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create D3D12 device");
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    qd.NodeMask = 0;

    hr = d3d_device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&d3d_queue_));
    if (FAILED(hr)) {
      printf("Failed to create command queue: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create command queue");
    }
    hr = DMLCreateDevice(d3d_device_.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                         IID_PPV_ARGS(&dml_device_));
    if (FAILED(hr)) {
      printf("Failed to create DML device: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create DML device");
    }

    OrtStatusPtr status = nullptr;
    status = Ort::GetApi().GetExecutionProviderApi(
        "DML", ORT_API_VERSION, reinterpret_cast<const void **>(&dml_api_));
    if (status != nullptr) {
      printf("Failed to create GPU allocation for features\n");
      throw std::runtime_error("Failed to get DML execution provider API");
    }

    printf(
        "D3D12 device, command queue, DML device, and DML API created "
        "successfully\n");
  }

  DmlMem AllocateDmlMem(size_t bytes) {
    DmlMem mem;
    ComPtr<ID3D12Resource> output_state_res = CreateUAVBuffer(bytes);
    if (!output_state_res) {
      printf("Failed to create output state resource\n");
      throw std::runtime_error("Failed to create output state resource");
    }

    void *output_state_dml_alloc = nullptr;
    OrtStatusPtr status = dml_api_->CreateGPUAllocationFromD3DResource(
        output_state_res.Get(), &output_state_dml_alloc);
    if (status != nullptr) {
      printf("Failed to create GPU allocation for output state\n");
      throw std::runtime_error(
          "Failed to create GPU allocation for output state");
    }
    return DmlMem{output_state_res, output_state_dml_alloc};
  }

  void FreeDmlMem(DmlMem &mem) {
    if (mem.data_) {
      dml_api_->FreeGPUAllocation(mem.data_);
      mem.data_ = nullptr;
    }
    mem.d2d_res_.Reset();
    mem.d2d_res_ = nullptr;
  }

  void CopyToGPU(const void *cpu_data, DmlMem *mem, size_t bytes,
                 const size_t &offset = 0) {
    void *gpu_alloc = mem->data_;
    ComPtr<ID3D12Resource> gpu_resource = mem->d2d_res_;
    // Check device removed status first
    HRESULT hr = d3d_device_->GetDeviceRemovedReason();
    if (FAILED(hr)) {
      printf("Device removed during CopyToGPU: 0x%08X\n", hr);
      throw std::runtime_error("Device removed during CopyToGPU");
    }

    // Validate parameters
    if (!cpu_data || !gpu_alloc || !gpu_resource || bytes == 0) {
      printf("Invalid parameters in CopyToGPU\n");
      throw std::runtime_error("Invalid parameters in CopyToGPU");
    }

    ComPtr<ID3D12Resource> upload_buffer;
    D3D12_HEAP_PROPERTIES upload_hp = {D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC upload_rd = {};
    upload_rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    upload_rd.Width = bytes;
    upload_rd.Height = 1;
    upload_rd.DepthOrArraySize = 1;
    upload_rd.MipLevels = 1;
    upload_rd.SampleDesc.Count = 1;
    upload_rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    hr = d3d_device_->CreateCommittedResource(
        &upload_hp, D3D12_HEAP_FLAG_NONE, &upload_rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&upload_buffer));
    if (FAILED(hr)) {
      printf("Failed to create upload buffer: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create upload buffer");
    }

    void *mapped_data;
    hr = upload_buffer->Map(0, nullptr, &mapped_data);
    if (FAILED(hr)) {
      printf("Failed to map upload buffer: 0x%08X\n", hr);
      throw std::runtime_error("Failed to map upload buffer");
    }
    memcpy(mapped_data, cpu_data, bytes);
    upload_buffer->Unmap(0, nullptr);

    ComPtr<ID3D12CommandAllocator> cmd_allocator;
    hr = d3d_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(&cmd_allocator));
    if (FAILED(hr)) {
      printf("Failed to create command allocator: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create command allocator");
    }

    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    hr = d3d_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        cmd_allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&cmd_list));
    if (FAILED(hr)) {
      printf("Failed to create command list: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create command list");
    }

    // Add resource barriers to ensure proper state transitions
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = gpu_resource.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list->ResourceBarrier(1, &barrier);

    // ʹ�ü������ƫ�������п���
    cmd_list->CopyBufferRegion(gpu_resource.Get(), offset, upload_buffer.Get(),
                               0, bytes);

    // Transition back to UAV state
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmd_list->ResourceBarrier(1, &barrier);

    hr = cmd_list->Close();
    if (FAILED(hr)) {
      printf("Failed to close command list: 0x%08X\n", hr);
      throw std::runtime_error("Failed to close command list");
    }

    ID3D12CommandList *cmd_lists[] = {cmd_list.Get()};
    d3d_queue_->ExecuteCommandLists(1, cmd_lists);

    ComPtr<ID3D12Fence> fence;
    hr = d3d_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
      printf("Failed to create fence: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create fence");
    }

    hr = d3d_queue_->Signal(fence.Get(), 1);
    if (FAILED(hr)) {
      printf("Failed to signal fence: 0x%08X\n", hr);
      throw std::runtime_error("Failed to signal fence");
    }

    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) {
      printf("Failed to create event\n");
      throw std::runtime_error("Failed to create event");
    }

    hr = fence->SetEventOnCompletion(1, event);
    if (FAILED(hr)) {
      printf("Failed to set event on completion: 0x%08X\n", hr);
      CloseHandle(event);
      throw std::runtime_error("Failed to set event on completion");
    }

    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
  }

  void CopyFromGPU(DmlMem *mem, void *cpu_data, size_t bytes,
                   const size_t &offset = 0) {
    void *gpu_alloc = mem->data_;
    ComPtr<ID3D12Resource> gpu_resource = mem->d2d_res_;

    // Check device removed status first
    HRESULT hr = d3d_device_->GetDeviceRemovedReason();
    if (FAILED(hr)) {
      printf("Device removed during CopyFromGPU: 0x%08X\n", hr);
      throw std::runtime_error("Device removed during CopyFromGPU");
    }

    ComPtr<ID3D12Resource> readback_buffer;
    D3D12_HEAP_PROPERTIES readback_hp = {D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC readback_rd = {};
    readback_rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_rd.Width = bytes;
    readback_rd.Height = 1;
    readback_rd.DepthOrArraySize = 1;
    readback_rd.MipLevels = 1;
    readback_rd.SampleDesc.Count = 1;
    readback_rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    hr = d3d_device_->CreateCommittedResource(
        &readback_hp, D3D12_HEAP_FLAG_NONE, &readback_rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback_buffer));
    if (FAILED(hr)) {
      printf("Failed to create readback buffer: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create readback buffer");
    }

    ComPtr<ID3D12CommandAllocator> cmd_allocator;
    hr = d3d_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(&cmd_allocator));
    if (FAILED(hr)) {
      printf("Failed to create command allocator: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create command allocator");
    }

    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    hr = d3d_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        cmd_allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&cmd_list));
    if (FAILED(hr)) {
      printf("Failed to create command list: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create command list");
    }

    // Add resource barriers to ensure proper state transitions
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = gpu_resource.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list->ResourceBarrier(1, &barrier);

    // ʹ�ü������ƫ�������п���
    cmd_list->CopyBufferRegion(readback_buffer.Get(), 0, gpu_resource.Get(),
                               offset, bytes);

    // Transition back to UAV state
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmd_list->ResourceBarrier(1, &barrier);

    cmd_list->Close();

    ID3D12CommandList *cmd_lists[] = {cmd_list.Get()};
    d3d_queue_->ExecuteCommandLists(1, cmd_lists);

    ComPtr<ID3D12Fence> fence;
    d3d_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    d3d_queue_->Signal(fence.Get(), 1);

    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(1, event);
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);

    void *mapped_data;
    readback_buffer->Map(0, nullptr, &mapped_data);
    memcpy(cpu_data, mapped_data, bytes);
    readback_buffer->Unmap(0, nullptr);
  }

  void CopyFromGPUToGPU(DmlMem *src_mem, DmlMem *dst_mem, size_t bytes,
                        const size_t &src_offset = 0,
                        const size_t &dst_offset = 0) {
    // Check device removed status first
    HRESULT hr = d3d_device_->GetDeviceRemovedReason();
    if (FAILED(hr)) {
      printf("Device removed during CopyGPUToGPU: 0x%08X\n", hr);
      throw std::runtime_error("Device removed during CopyGPUToGPU");
    }

    // Validate parameters
    if (!src_mem || !dst_mem || !src_mem->data_ || !dst_mem->data_ ||
        !src_mem->d2d_res_ || !dst_mem->d2d_res_ || bytes == 0) {
      printf("Invalid parameters in CopyGPUToGPU\n");
      throw std::runtime_error("Invalid parameters in CopyGPUToGPU");
    }

    ComPtr<ID3D12Resource> src_resource = src_mem->d2d_res_;
    ComPtr<ID3D12Resource> dst_resource = dst_mem->d2d_res_;

    ComPtr<ID3D12CommandAllocator> cmd_allocator;
    hr = d3d_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(&cmd_allocator));
    if (FAILED(hr)) {
      printf("Failed to create command allocator: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create command allocator");
    }

    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    hr = d3d_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        cmd_allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&cmd_list));
    if (FAILED(hr)) {
      printf("Failed to create command list: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create command list");
    }

    // Set up resource barriers for proper state transitions
    D3D12_RESOURCE_BARRIER barriers[2] = {};

    // Source transition: UAV -> COPY_SOURCE
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barriers[0].Transition.pResource = src_resource.Get();
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[0].Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    // Destination transition: UAV -> COPY_DEST
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barriers[1].Transition.pResource = dst_resource.Get();
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[1].Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmd_list->ResourceBarrier(2, barriers);

    
    cmd_list->CopyBufferRegion(dst_resource.Get(), dst_offset,
                               src_resource.Get(), src_offset, bytes);

    // Transition back to UAV state
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    cmd_list->ResourceBarrier(2, barriers);

    hr = cmd_list->Close();
    if (FAILED(hr)) {
      printf("Failed to close command list: 0x%08X\n", hr);
      throw std::runtime_error("Failed to close command list");
    }

    // Execute the command list
    ID3D12CommandList *cmd_lists[] = {cmd_list.Get()};
    d3d_queue_->ExecuteCommandLists(1, cmd_lists);

    // Wait for completion
    ComPtr<ID3D12Fence> fence;
    hr = d3d_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
      printf("Failed to create fence: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create fence");
    }

    hr = d3d_queue_->Signal(fence.Get(), 1);
    if (FAILED(hr)) {
      printf("Failed to signal fence: 0x%08X\n", hr);
      throw std::runtime_error("Failed to signal fence");
    }

    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (event == nullptr) {
      printf("Failed to create event\n");
      throw std::runtime_error("Failed to create event");
    }

    hr = fence->SetEventOnCompletion(1, event);
    if (FAILED(hr)) {
      printf("Failed to set event on completion: 0x%08X\n", hr);
      CloseHandle(event);
      throw std::runtime_error("Failed to set event on completion");
    }

    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
  }

  void WaitForGPU() {
    static uint64_t fence_value = 0;

    // Check device status before creating fence
    HRESULT hr = d3d_device_->GetDeviceRemovedReason();
    if (FAILED(hr)) {
      printf("Device removed in WaitForGPU: 0x%08X\n", hr);
      throw std::runtime_error("Device removed in WaitForGPU");
    }

    ComPtr<ID3D12Fence> fence;
    hr = d3d_device_->CreateFence(fence_value, D3D12_FENCE_FLAG_NONE,
                                  IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
      printf("Failed to create fence: 0x%08X\n", hr);
      throw std::runtime_error("Failed to create fence");
    }

    fence_value++;
    hr = d3d_queue_->Signal(fence.Get(), fence_value);
    if (FAILED(hr)) {
      printf("Failed to signal fence in WaitForGPU: 0x%08X\n", hr);
      throw std::runtime_error("Failed to signal fence in WaitForGPU");
    }

    if (fence->GetCompletedValue() < fence_value) {
      HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
      if (event == nullptr) {
        printf("Failed to create event in WaitForGPU\n");
        throw std::runtime_error("Failed to create event in WaitForGPU");
      }

      hr = fence->SetEventOnCompletion(fence_value, event);
      if (FAILED(hr)) {
        printf("Failed to set event on completion in WaitForGPU: 0x%08X\n", hr);
        CloseHandle(event);
        throw std::runtime_error(
            "Failed to set event on completion in WaitForGPU");
      }

      DWORD wait_result = WaitForSingleObject(event, 5000);  // 5 second timeout
      if (wait_result != WAIT_OBJECT_0) {
        printf("GPU wait timeout or failed, result: %d\n", wait_result);
      }
      CloseHandle(event);
    }
  }

  Ort::SessionOptions CreateSessionOptions() {
    Ort::SessionOptions session_options;
    session_options.DisableMemPattern();
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    dml_api_->SessionOptionsAppendExecutionProvider_DML1(
        session_options, dml_device_.Get(), d3d_queue_.Get());
    return session_options;
  }

  void FlushCommandLists() { d3d_queue_->ExecuteCommandLists(0, nullptr); }

 private:
  ComPtr<ID3D12Resource> CreateUAVBuffer(uint64_t bytes) {
    ComPtr<ID3D12Resource> res;
    D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    HRESULT hr = d3d_device_->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr, IID_PPV_ARGS(&res));

    if (FAILED(hr)) {
      printf("Failed to create UAV buffer of size %llu: 0x%08X\n", bytes, hr);
      return nullptr;
    }

    return res;
  }

 private:
  // D3D12 device and command queue
  ComPtr<ID3D12Device> d3d_device_;
  ComPtr<ID3D12CommandQueue> d3d_queue_;
  ComPtr<IDMLDevice> dml_device_;
  const OrtDmlApi *dml_api_ = nullptr;
};

}  // namespace sherpa_onnx

#endif  // HYLLO_DML_UTILS_H