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
    const int* di_data = (int*) input_mat_vec[1]->gpu_data();

    const size_t running_batch_size = (trailing_m_ == 0 )?m_:trailing_m_;

    //run indexed sum
    gk_isum(running_batch_size, n_, k_, x_data, di_data, output_mat_vec[0]->mutable_gpu_data(), output_mat_vec[1]->mutable_gpu_data(), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream)); // seem to be causing trouble if we don't do this.
  }

  INSTANTIATE_CLASS(CenterOfMassFunction);

}