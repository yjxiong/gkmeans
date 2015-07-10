//
// Created by alex on 7/10/15.
//

#ifndef GKMEANS_MAT_H
#define GKMEANS_MAT_H

#include "gkmeans/common.h"
#include "gkmeans/mem.h"

namespace gkmeans{

  /**
   * @brief Class for holding matrix and array
   */
  template <typename Dtype>
  class Mat{
  public:
    Mat(vector<size_t> shape, int device_id);
    Mat():Mat(vector<size_t>({0})){};
    Mat(vector<size_t> shape) : Mat(shape, 0){}

    // shape information accessors
    const size_t count(){return count_;};
    const vector<size_t>& shape(){ return shape_;};
    const size_t shape(int dim){return shape_[dim];}

    void Reshape(vector<size_t> shape);

    /**
     * @brief conduct memory copy from host to GPU
     */
    inline void to_gpu_async(cudaStream_t stream){mem_->to_gpu_async(stream);};

    /**
     * @brief conduct memory copy from GPU to GPU
     */
    inline void to_cpu_async(cudaStream_t stream){mem_->to_cpu_async(stream);};

    // Data accessors
    inline const Dtype* gpu_data(){return static_cast<const Dtype*>(mem_->gpu_data());};
    inline Dtype* mutable_gpu_data(){return static_cast<Dtype*>(mem_->mutable_gpu_data());};
    inline const Dtype* cpu_data(){return static_cast<const Dtype*>(mem_->cpu_data());};
    inline Dtype* mutable_cpu_data(){return static_cast<Dtype*>(mem_->mutable_cpu_data());};

    inline const int device_id(){return device_id_;}

  protected:
    shared_ptr<Mem> mem_;
    vector<size_t> shape_;
    size_t count_;
    int device_id_;
  };
}

#endif //GKMEANS_MAT_H
