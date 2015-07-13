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

    CHECK_EQ(input_mat_vec.size(), 3);
    CHECK_EQ(output_mat_vec.size(), 1);

    m_ = input_mat_vec[0]->shape(0);
    k_ = input_mat_vec[0]->shape(1);
    CHECK_EQ(input_mat_vec[1]->count(), m_);
    n_ = (int)input_mat_vec[2]->cpu_data()[0];

    // shape row index buffer
    buffer_row_idx_.reset(new Mat<Dtype>(vector<size_t>({m_ + 1,1})));
    buffer_transpose_.reset(new Mat<Dtype>(vector<size_t>({k_, n_})));
    buffer_trans_row_idx_.reset(new Mat<Dtype>(vector<size_t>({m_ + 1})));
    buffer_trans_col_idx_.reset(new Mat<Dtype>(vector<size_t>({m_ + 1})));
    buffer_ones_.reset(new Mat<Dtype>(vector<size_t>({m_ + 1})));
    buffer_trans_Y_.reset(new Mat<Dtype>(vector<size_t>({n_, k_})));
    buffer_isum_.reset(new Mat<Dtype>(vector<size_t>({n_})));

    int* row_index_data = (int*) buffer_row_idx_->mutable_cpu_data();
    for (size_t i = 0; i < m_ + 1; i++ ){
      row_index_data[i] = int(i);
    }
    buffer_row_idx_->gpu_data();

    // shape output blob
    output_mat_vec[0]->Reshape(vector<size_t>({n_, k_}));

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
    Dtype* y_data = output_mat_vec[0]->mutable_gpu_data();
    Dtype* isum_data = buffer_isum_->mutable_gpu_data();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start, stream);
    gk_isum(m_, n_, k_, x_data, di_data, y_data, isum_data, stream);


    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    cout <<"update time: "<<ms<<" ms\n";
  }

  template<typename Dtype>
  void CenterOfMassFunction<Dtype>::cusparseExecute(
      const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec, cudaStream_t stream
  ){
    int* cscRowInd = (int*)buffer_trans_row_idx_->mutable_gpu_data();
    int* cscColPtr = (int*)buffer_trans_col_idx_->mutable_gpu_data();
    Dtype* cscData = buffer_ones_->mutable_gpu_data();

    int* csrRowPtr = (int*) buffer_row_idx_->mutable_gpu_data();
    int* csrColInd = (int*) input_mat_vec[1]->mutable_gpu_data();

    const Dtype* x_data = input_mat_vec[0]->gpu_data();
    Dtype* trans_y_data = buffer_trans_Y_->mutable_gpu_data();
    Dtype* y_data = output_mat_vec[0]->mutable_gpu_data();

    //actually we are transposing the assignment matrix
    gk_csr2csc<Dtype>(m_, n_, m_, cscData, csrRowPtr, csrColInd, cscData, cscRowInd, cscColPtr, stream);

    gk_sparse_gemm2<Dtype>(CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    n_, k_, m_, m_, Dtype(1), cscData, cscColPtr, cscRowInd, x_data, Dtype(0), trans_y_data, stream);

  }

  INSTANTIATE_CLASS(CenterOfMassFunction);

}