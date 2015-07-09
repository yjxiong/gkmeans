//
// Created by alex on 7/9/15.
//

#ifndef GKMEANS_MEM_H
#define GKMEANS_MEM_H

#include "gkmeans/common.h"

namespace gkmeans{

  /**
   * @brief the class for supporting GPU and CPU memory
   * This class allows async memory copy between host and device.
   * Thus we can use a "double buffer" like operation to overlap memory transfer and compuation
   */
  class Mem{
  public:
    Mem(size_t count);
    Mem(): Mem(0){}

    enum HEAD {
      GPU,
      CPU,
      SYNCED,
      NOT_INITIALIZED
    };

    const void* cpu_data();
    const void* gpu_data();
    void* mutable_cpu_data();
    void* mutable_gpu_data();

    void to_gpu_async(cudaStream_t stream);
    void to_gpu_sync();
    void to_cpu_async(cudaStream_t stream);
    void to_cpu_sync();

    inline const size_t count(){return count_;}
    void resize(size_t new_count);

  protected:
    size_t count_;
    HEAD head_at_;

    void* gpu_mem_;
    void* cpu_mem_;

    cudaStream_t transfer_stream_;

  private:
    inline void init_gpu_mem(){
      if (!gpu_mem_ && count_){
        CUDA_CHECK(cudaMalloc(&gpu_mem_, count_));
        cudaMemset(gpu_mem_, 0, count_);
      }
    }
    inline void init_cpu_mem(){
      if (!cpu_mem_ && count_){
        CUDA_CHECK(cudaMallocHost(&cpu_mem_, count_));
        memset(cpu_mem_, 0, count_);
      }
    }

    inline void wait_transfer(){
      if (transfer_stream_){
        CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
      }
    }
  };
}

#endif //GKMEANS_MEM_H
