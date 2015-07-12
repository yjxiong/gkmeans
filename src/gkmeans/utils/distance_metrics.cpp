//
// Created by alex on 7/10/15.
//


#include "gkmeans/common.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/distance_metrics.h"

namespace gkmeans{
  template<typename Dtype>
  void gk_euclidean_dist(int M, int N, int k, const Dtype* X, const Dtype* Y, Dtype* D,
                         Dtype* buffer_X2, Dtype* buffer_Y2,
                         const Dtype* buffer_ones, Dtype* buffer_norm,
                         cudaStream_t stream){

    // calculte the 2 norm of each element
    gk_pow<Dtype>(M * k, X, Dtype(2), buffer_X2, stream);
    gk_pow<Dtype>(N * k, Y, Dtype(2), buffer_Y2, stream);

    // calculate X * Y^T
    gk_gemm<Dtype>(CUBLAS_OP_N, CUBLAS_OP_T, M, N, k, -2.0, X, Y,  0., D, stream);

    // add norm(Y,2)
    gk_gemv<Dtype>(CUBLAS_OP_N, N, k, 1.0, buffer_Y2, buffer_ones, 0., buffer_norm, stream);
    gk_ger<Dtype>(M, N, buffer_ones, buffer_norm, Dtype(1), D, stream);

    // add norm(X, 2)
    gk_gemv<Dtype>(CUBLAS_OP_N, M, k, 1.0, buffer_X2, buffer_ones, 0., buffer_norm, stream);
    gk_ger<Dtype>(M, N, buffer_norm, buffer_ones, Dtype(1), D, stream);
  }

  template void gk_euclidean_dist<float>(int M, int N, int k, const float* X, const float* Y, float* D,
                                  float* buffer_X2, float* buffer_Y2,
                                  const float* buffer_ones, float* buffer_norm,
                                  cudaStream_t stream);

  template<typename Dtype>
  void gk_shortest_euclidean_dist(int M, int N, int k, const Dtype* X,
                                  const Dtype* Y, int* DI, Dtype* D,
                                  Dtype* buffer_X2, Dtype* buffer_Y2,
                                  const Dtype* buffer_ones, Dtype* buffer_XY, Dtype* buffer_norm,
                                  cudaStream_t stream){

    gk_pow(M * k , X, Dtype(2), buffer_X2, stream);
    gk_pow(N * k, Y, Dtype(2), buffer_Y2, stream);

    //calculate norm of Y for rank-1 update
    gk_gemv<Dtype>(CUBLAS_OP_N, N, k, 1.0, buffer_Y2, buffer_ones, 0., buffer_norm, stream);

    // conduct gemm, reverse the order of Y and X so that the result is in a column-major format
    gk_gemm(CUBLAS_OP_N, CUBLAS_OP_T, N, M, k, Dtype(-2), Y, X,  Dtype(0.), buffer_XY, stream);
    gk_ger(N, M, buffer_norm, buffer_ones, Dtype(1), buffer_XY, stream);

    //calculate the min for each row in the matrix
    //This function inputs column-major matrix to achieve coalesced memory access
    gk_rmin(M, N, buffer_XY, DI, D, stream);

    //finally put the norms into the slots
    gk_gemv<Dtype>(CUBLAS_OP_N, M, k, 1.0, buffer_X2, buffer_ones, 1., D, stream);

  }

  template void gk_shortest_euclidean_dist<float>(int M, int N, int k, const float* X, const float* Y, int* DI, float* D,
                                                  float* buffer_X2, float* buffer_Y2,
                                                  const float* buffer_ones, float* buffer_XY, float* buffer_norm,
                                                  cudaStream_t stream);
}
