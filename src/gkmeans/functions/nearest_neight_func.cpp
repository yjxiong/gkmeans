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
  void NearestNeighborFunction<Dtype>::FunctionSetUp(
      const vector<Mat<Dtype> *> &input_mat_vec, const vector<Mat<Dtype> *> &output_mat_vec) {

    Mat<Dtype>* X_mat = input_mat_vec[0];
    Mat<Dtype>* Y_mat = input_mat_vec[1];

    CHECK_EQ(X_mat->shape().size(), 2);

    m_ = X_mat->shape(0);
    n_ = Y_mat->shape(0);

    CHECK_EQ(X_mat->shape(1), Y_mat->shape(1));
    k_ = X_mat->shape(1);

    max_num_ = std::max(m_, n_);
    max_dim_ = std::max(max_num_, k_);

    //setup buffers
    buffer_X2_.reset(new Mat<Dtype>(vector<size_t>({m_, k_})));
    buffer_Y2_.reset(new Mat<Dtype>(vector<size_t>({n_, k_})));
    buffer_XY_.reset(new Mat<Dtype>(vector<size_t>({m_, n_})));
    buffer_ones_.reset(new Mat<Dtype>(vector<size_t>({max_dim_})));
    buffer_norm_.reset(new Mat<Dtype>(vector<size_t>({max_num_})));

    Dtype* ones_data = buffer_ones_->mutable_cpu_data();
    for (size_t i = 0; i < max_dim_; ++i){
      ones_data[i] = Dtype(1.);
    }
    buffer_ones_->gpu_data();
    //setup output mats

    output_mat_vec[0]->Reshape(vector<size_t>({m_}));
    output_mat_vec[1]->Reshape(vector<size_t>({m_}));
  }

  template<typename Dtype>
  void NearestNeighborFunction<Dtype>::Execute(
      const vector<Mat<Dtype> *> &input_mat_vec, const vector<Mat<Dtype> *> &output_mat_vec,
      cudaStream_t stream) {

    const Dtype* x_data = input_mat_vec[0]->gpu_data();
    const Dtype* y_data = input_mat_vec[1]->gpu_data();

    Dtype* D_data = output_mat_vec[1]->mutable_gpu_data();
    int* DI_data = (int*)output_mat_vec[0]->mutable_gpu_data();

    //buffer data
    Dtype* x2_data = buffer_X2_->mutable_gpu_data();
    Dtype* y2_data = buffer_Y2_->mutable_gpu_data();
    const Dtype* ones_data = buffer_ones_->gpu_data();
    Dtype* norm_data = buffer_norm_->mutable_gpu_data();
    Dtype* xy_data = buffer_XY_->mutable_gpu_data();

    //Conduct shortest distance search
    gk_shortest_euclidean_dist<Dtype>(m_, n_, k_, x_data, y_data, DI_data, D_data,
                                      x2_data, y2_data, ones_data, xy_data, norm_data, stream);
  }

  INSTANTIATE_CLASS(NearestNeighborFunction);

}