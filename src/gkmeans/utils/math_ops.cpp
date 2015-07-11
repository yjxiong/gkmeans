//
// Created by alex on 7/10/15.
//

#include "gkmeans/common.h"
#include "gkmeans/utils/cuda_utils.h"
#include "gkmeans/utils/math_ops.h"

namespace gkmeans{

  template <>
  void gk_scale<float>(const int N, const float alpha, float* data, cudaStream_t stream){
    CUBLAS_SET_STREAM(stream);
    CUBLAS_CHECK(cublasSscal(GKMeans::cublas_handle(), N, &alpha, data, 1));
  }

  template<>
  void gk_axpby<float>(const int N, const float alpha, const float *X,
                       const float beta, float *Y,
                       cudaStream_t stream){
    if (beta != 1.0){
      gk_scale(N, beta, Y, stream);
    }
    CUBLAS_SET_STREAM(stream);
    CUBLAS_CHECK(cublasSaxpy(GKMeans::cublas_handle(), N, &alpha,
                             X, 1, Y, 1));
  }



  template <>
  void gk_gemm<float>(const cublasOperation_t TransA,
               const cublasOperation_t TransB, const int M, const int N, const int K,
               const float alpha, const float* A, const float* B, const float beta,
               float* C,
               cudaStream_t stream){
    // Note that cublas follows fortran order.
    int lda = (TransA == CUBLAS_OP_N) ? K : M;
    int ldb = (TransB == CUBLAS_OP_N) ? N : K;
    CUBLAS_SET_STREAM(stream);
    CUBLAS_CHECK(cublasSgemm(GKMeans::cublas_handle(), TransB, TransA,
                             N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));

  }

  template <>
  void gk_gemv<float>(const cublasOperation_t TransA, const int M, const int N,
                      const float alpha, const float* A, const float* x, const float beta,
                      float* y,
               cudaStream_t stream){
    cublasOperation_t cuTransA =
        (TransA == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_SET_STREAM(stream);
    CUBLAS_CHECK(cublasSgemv(GKMeans::cublas_handle(), cuTransA, N, M, &alpha,
                             A, N, x, 1, &beta, y, 1));

  }

  /** ger (outer product) with CUBLAS */
  template<>
  void gk_ger<float>(const int M, const int N,
                     const float *x, const float* y,
                     const float alpha,
                     float *A, cudaStream_t stream){
    CUBLAS_SET_STREAM(stream);
    CUBLAS_CHECK(cublasSger(GKMeans::cublas_handle(), N, M, &alpha, y, 1, x, 1, A, N ));
  }
}
