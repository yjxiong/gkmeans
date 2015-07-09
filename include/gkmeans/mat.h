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
    Mat(vector<int> shape);
    Mat():Mat(vector<size_t>({0})){};

    // shape information accessors
    const size_t count(){return count_;};
    const vector<size_t>& shape(){ return shape_;};
    const size_t shape(int dim){return shape_[dim];}

    void reshape(vector<size_t> shape);

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
    inline Dtype* mutable_gpu_data(){return static_cast<Dtype*>(mem_->gpu_data());};
    inline const Dtype* cpu_data(){return static_cast<const Dtype*>(mem_->gpu_data());};
    inline Dtype* mutable_cpu_data(){return static_cast<Dtype*>(mem_->gpu_data());};

  protected:
    shared_ptr<Mem> mem_;
    vector<size_t> shape_;
    size_t count_;
  };
}

#endif //GKMEANS_MAT_H
