//
// Created by alex on 7/12/15.
//

#include "gkmeans/common.h"
#include "gkmeans/functions.h"
#include "gkmeans/comp_functions.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/distance_metrics.h"
#include "gkmeans/mat.h"

namespace gkmeans{

  template<typename Dtype>
  void CenterOfMassFunction<Dtype>::FunctionSetUp(
      const vector<Mat<Dtype> *> &input_mat_vec, const vector<Mat<Dtype> *> &output_mat_vec) {

    m_ = input_mat_vec[0]->shape(0);
    k_ = input_mat_vec[0]->shape(1);
    CHECK_EQ(input_mat_vec[1]->count(), m_);
    n_ = std::stoul(GKMeans::get_config("n_cluster"));

    // shape the local buffer
    buffer_y_.reset(new Mat<Dtype>(vector<size_t>({n_, k_})));
    buffer_isum_.reset(new Mat<Dtype>(vector<size_t>({n_})));

    // shape output blob
    output_mat_vec[0]->Reshape(vector<size_t>({n_, k_}));
    output_mat_vec[1]->Reshape(vector<size_t>({n_}));

  }

  template<typename Dtype>
  void CenterOfMassFunction<Dtype>::Execute(
      const vector<Mat<Dtype> *> &input_mat_vec, const vector<Mat<Dtype> *> &output_mat_vec,
      cudaStream_t stream) {

    kernelExecute(input_mat_vec, output_mat_vec, stream);
  }

  template<typename Dtype>
  void CenterOfMassFunction<Dtype>::kernelExecute(
      const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec, cudaStream_t stream
  ){
    const Dtype* x_data = input_mat_vec[0]->gpu_data();
    const int* di_data = (int*)input_mat_vec[1]->gpu_data();
    Dtype* y_data = buffer_y_->mutable_gpu_data();
    Dtype* isum_data = buffer_isum_->mutable_gpu_data();

    //run indexed sum
    gk_isum(m_, n_, k_, x_data, di_data, y_data, isum_data, stream);

    gk_axpby(buffer_y_->count(), Dtype(1), y_data, Dtype(1), output_mat_vec[0]->mutable_gpu_data(), stream);
    gk_axpby(buffer_isum_->count(), Dtype(1), isum_data, Dtype(1), output_mat_vec[1]->mutable_gpu_data(), stream);

  }

  INSTANTIATE_CLASS(CenterOfMassFunction);

}