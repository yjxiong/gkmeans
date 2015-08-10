//
// Created by alex on 7/9/15.
//

#ifndef GKMEANS_MEM_H
#define GKMEANS_MEM_H

#include "gkmeans/common.h"
#include "gkmeans/utils/cuda_utils.h"

namespace gkmeans{

  /**
   * @brief the class for supporting GPU and CPU memory
   * This class allows async memory copy between host and device.
   * Thus we can use a "double buffer" like operation to overlap memory transfer and compuation
   */
  class Mem{
  public:
    Mem(size_t count, int device_id);
    Mem(): Mem(0){}
    Mem(size_t count) : Mem(count, 0){}
    ~Mem();

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

    inline const int device_id(){return device_id_;}

  protected:
    size_t count_;
    int device_id_;
    HEAD head_at_;
    void* gpu_mem_;
    void* cpu_mem_;
    cudaStream_t transfer_stream_;

  private:
    inline void init_gpu_mem(bool zero=false){
      if (!gpu_mem_ && count_){
        int dev; cudaGetDevice(&dev);
        CUDA_CHECK(cudaMalloc(&gpu_mem_, count_));
        if (zero) CUDA_CHECK(cudaMemset(gpu_mem_, 0, count_));
      }
    }
    inline void init_cpu_mem(bool zero=false){
      if (!cpu_mem_ && count_){
        CUDA_CHECK(cudaMallocHost(&cpu_mem_, count_));
        if (zero) memset(cpu_mem_, 0, count_);
      }
    }

    inline void wait_transfer(){
      if (transfer_stream_){
        CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
        transfer_stream_ = NULL;
      }
    }
  };
}

#endif //GKMEANS_MEM_H
