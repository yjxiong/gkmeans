//
// Created by alex on 7/10/15.
//

#include "gkmeans/mat.h"
#include "gkmeans/utils/cuda_utils.h"

namespace gkmeans{

  template<typename Dtype>
  Mat<Dtype>::Mat(vector<size_t> shape, int device_id)
      : shape_(shape), device_id_(device_id) {

    // set count
    count_ = 1;
    for(auto s : shape) count_*=s;
    mem_.reset(new Mem(count_ * sizeof(Dtype), device_id_));
  }

  template<typename Dtype>
  void Mat<Dtype>::Reshape(vector<size_t> shape) {
    size_t new_cnt = 1;
    for (auto s: shape) new_cnt *= s;

    //do a new allocation when needed
    shape_ = shape;
    count_ = new_cnt;
    mem_.reset(new Mem(count_, device_id_));
  }

  template class Mat<int>;
  template class Mat<float>;

}
