//
// Created by alex on 7/9/15.
//

#include "gkmeans/mem.h"

namespace gkmeans{
  Mem::Mem(size_t count)
      : count_(count),
        head_at_(Mem::NOT_INITIALIZED),
        gpu_mem_(NULL), cpu_mem_(NULL),
        transfer_stream_(NULL) {}

  const void* Mem::cpu_data() {
    to_cpu_sync();
    return cpu_mem_;
  }

  const void* Mem::gpu_data() {
    to_gpu_sync();
    return gpu_mem_;
  }

  void* Mem::mutable_cpu_data() {
    to_cpu_sync();
    head_at_ = CPU;
    return cpu_mem_;
  }
  void* Mem::mutable_gpu_data() {
    to_gpu_sync();
    head_at_ = GPU;
    return gpu_mem_;
  }

  void Mem::to_cpu_async(cudaStream_t stream) {
    switch (head_at_){
      case CPU:{
        break;
      }
      case GPU:{
        if (!stream) {
          if (!cpu_mem_) init_cpu_mem();
          CUDA_CHECK(cudaMemcpyAsync(cpu_mem_, gpu_mem_, count_, cudaMemcpyDeviceToHost, stream));
          transfer_stream_ = stream;
          head_at_ = SYNCED;
        }
        break;
      }
      case SYNCED:
        break;
      case NOT_INITIALIZED:{
        init_cpu_mem();
        head_at_ = CPU;
      }
    }

  }

  void Mem::to_gpu_async(cudaStream_t stream) {
    switch (head_at_){
      case GPU:{
        break;
      }
      case CPU:{
        if (!stream) {
          if (!gpu_mem_) init_gpu_mem();
          CUDA_CHECK(cudaMemcpyAsync(gpu_mem_, cpu_mem_, count_, cudaMemcpyHostToDevice, stream));
          transfer_stream_ = stream;
          head_at_ = SYNCED;
        }
        break;
      }
      case SYNCED:
        break;
      case NOT_INITIALIZED:{
        init_cpu_mem();
        head_at_ = GPU;
      }
    }
  }

  void Mem::to_cpu_sync() {
    wait_transfer();
    switch (head_at_){
      case CPU:{
        break;
      }
      case GPU:{
        if (!cpu_mem_) init_cpu_mem();
        CUDA_CHECK(cudaMemcpy(cpu_mem_, gpu_mem_, count_, cudaMemcpyDeviceToHost));
        head_at_ = SYNCED;
        break;
      }
      case SYNCED:
        break;
      case NOT_INITIALIZED:{
        init_cpu_mem();
        head_at_ = CPU;
      }
    }
  }

  void Mem::to_gpu_sync() {
    wait_transfer();
    switch (head_at_){
      case GPU:{
        break;
      }
      case CPU:{
        if (!gpu_mem_) init_gpu_mem();
        CUDA_CHECK(cudaMemcpy(cpu_mem_, gpu_mem_, count_, cudaMemcpyHostToDevice));
        head_at_ = SYNCED;
        break;
      }
      case SYNCED:
        break;
      case NOT_INITIALIZED:{
        init_gpu_mem();
        head_at_ = GPU;
      }
    }
  }
}