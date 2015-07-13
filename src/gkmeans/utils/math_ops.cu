//
// Created by alex on 7/10/15.
//

#include "gkmeans/common.h"
#include "gkmeans/utils/cuda_utils.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/cuda_utils.h"
#include "../../../include/gkmeans/utils/cuda_utils.h"

namespace gkmeans{
  template <typename Dtype>
  __global__ void scaler_add_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] = a[index] + b[index];
    }
  }

  template<typename Dtype>
  void gk_add(const int N, const Dtype *a, const Dtype *b, Dtype *y, cudaStream_t stream){
    // NOLINT_NEXT_LINE(whitespace/operators)
    scaler_add_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
        N, a, b, y);
    CUDA_POST_KERNEL_CHECK;
  }

  template void gk_add<float>(const int N, const float *a, const float *b, float *y, cudaStream_t stream);

  template <typename Dtype>
  __global__ void scaler_pow_kernel(const int n, const Dtype *x, const Dtype alpha, Dtype *y);

  template <>
  __global__ void scaler_pow_kernel<float>(const int n, const float *x, const float alpha, float *y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] = powf(x[index], alpha);
    }
  }

  template <>
  __global__ void scaler_pow_kernel<double>(const int n, const double *x, const double alpha, double *y) {
    CUDA_KERNEL_LOOP(index, n) {
      y[index] = pow(x[index], alpha);
    }
  }

  template<typename Dtype>
  void gk_pow(const int N, const Dtype *X, const Dtype alpha, Dtype *Y, cudaStream_t stream){
    scaler_pow_kernel<Dtype><<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
        N, X, alpha, Y);
    CUDA_POST_KERNEL_CHECK;
  }

  template void gk_pow<float>(const int N, const float *X, const float alpha, float *Y, cudaStream_t stream);
  template void gk_pow<double>(const int N, const double *X, const double alpha, double *Y, cudaStream_t stream);


  template <typename Dtype>
  __global__ void rmin_kernel(const int M, const int N, const Dtype* data, int* max_idx, Dtype* max_val) {
    CUDA_KERNEL_LOOP(row, M) {
      Dtype v_min = data[row];
      Dtype idx_min = 0;

      //scan for max
      for (int j = 0; j < N; ++j){
        const Dtype val = data[j * M + row];
        if ( val< v_min){
          v_min = val;
          idx_min = j;
        }
      }

      //assign val to output
      max_idx[row] = idx_min;
      max_val[row] = v_min;
    }
  }

  /** @per row argmax and max with CUDA
   *  Note: This function requires column major input!
   * */
  template <typename Dtype>
  void gk_rmin(const int M, const int N, const Dtype* data, int* max_idx, Dtype* max_val, cudaStream_t stream){
    rmin_kernel<Dtype><<<CUDA_GET_BLOCKS(M), CUDA_NUM_THREADS, 0, stream>>>(
        M, N, data, max_idx, max_val);
    CUDA_POST_KERNEL_CHECK;
  }

  template void gk_rmin<float>(const int M, const int N, const float* data, int* max_idx, float* max_val, cudaStream_t stream);


  template <typename Dtype>
  __global__ void isum_kernel(const int M, const int N, const int K, const Dtype* X, const int* DI, Dtype* Y, Dtype* ISum){
    CUDA_KERNEL_LOOP(dim, K) {
      for (int row = 0; row < M; ++row){
        int data_index = row * K + dim;
        int index = DI[row];
        Y[index * K + dim] += X[data_index];

        if (dim == 0) ISum[index] += 1;
      }
    }
  }

  /** indexed sum **/
  template <typename Dtype>
  void gk_isum(const int M, const int N, const int K, const Dtype* X, const int* DI,
               Dtype* Y, Dtype* ISum, cudaStream_t stream){
    isum_kernel<<<CUDA_GET_BLOCKS(K), CUDA_NUM_THREADS, 0, stream>>>(M, N, K, X, DI, Y, ISum);
    CUDA_POST_KERNEL_CHECK;
  }

  template void gk_isum<float>(const int M, const int N, const int K, const float* X, const int* DI,
                               float* Y, float* ISum, cudaStream_t stream);
}