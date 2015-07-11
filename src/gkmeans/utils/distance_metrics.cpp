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
    gk_pow<Dtype>(M * k, X, Dtype(2), buffer_X2, stream);
    gk_pow<Dtype>(N * k, Y, Dtype(2), buffer_Y2, stream);
    gk_gemm<Dtype>(CUBLAS_OP_N, CUBLAS_OP_T, M, N, k, -2.0, X, Y,  0., D, stream);
    gk_gemv<Dtype>(CUBLAS_OP_N, N, k, 1.0, buffer_Y2, buffer_ones, 0., buffer_norm, stream);
    gk_ger<Dtype>(M, N, buffer_ones, buffer_norm, Dtype(1), D, stream);
    gk_gemv<Dtype>(CUBLAS_OP_N, M, k, 1.0, buffer_X2, buffer_ones, 0., buffer_norm, stream);
    gk_ger<Dtype>(M, N, buffer_norm, buffer_ones, Dtype(1), D, stream);
  }

  template void gk_euclidean_dist<float>(int M, int N, int k, const float* X, const float* Y, float* D,
                                  float* buffer_X2, float* buffer_Y2,
                                  const float* buffer_ones, float* buffer_norm,
                                  cudaStream_t stream);

  template<typename Dtype>
  void gk_shortest_euclidean_dist(int M, int N, int k, const Dtype* X, const Dtype* Y, Dtype* DI, Dtype* D,
                                  Dtype* buffer_X2, Dtype* buffer_Y2,
                                  Dtype* buffer_ones, Dtype* buffer_XY,
                                  cudaStream_t stream){
    gk_pow(M, X, Dtype(2), buffer_X2, stream);
    gk_pow(N, Y, Dtype(2), buffer_Y2, stream);
    gk_gemm(CUBLAS_OP_N, CUBLAS_OP_T, M, N, k, Dtype(-2), X, Y,  Dtype(0.), buffer_XY, stream);
    gk_ger(M, N, buffer_ones, buffer_Y2, Dtype(1), buffer_XY, stream);

    //TODO: write test for this
    gk_rmin(M, N, buffer_XY, DI, D, stream);

    gk_axpby(M, Dtype(1), buffer_X2, Dtype(1), D, stream);
  }
}
