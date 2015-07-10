//
// Created by alex on 7/10/15.
//

#include "gkmeans/common.h"
#include "gkmeans/utils/cuda_utils.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/cuda_utils.h"

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
  }

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
  }
}