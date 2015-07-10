//
// Created by alex on 7/10/15.
//


#include "gkmeans/common.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/distance_metrics.h"

namespace gkmeans{
  template<typename Dtype>
  void gk_euclidean_dist(int M, int N, int k, const Dtype* X, const Dtype* Y, Dtype* D,
                         Dtype* buffer_X2, Dtype* buffer_Y2, Dtype* buffer_ones,
                         cudaStream_t stream){
    gk_pow(M, X, Dtype(2), buffer_X2, stream);
    gk_pow(N, Y, Dtype(2), buffer_Y2, stream);
    gk_gemm(CUBLAS_OP_N, CUBLAS_OP_T, M, N, k, Dtype(-2), X, Y,  Dtype(0.), D, stream);
    gk_ger(M, N, buffer_ones, buffer_Y2, Dtype(1), D, stream);
    gk_ger(M, N, buffer_X2, buffer_ones, Dtype(1), D, stream);
  }

  template<typename Dtype>
  void gk_shortest_euclidean_dist(int M, int N, int k, const Dtype* X, const Dtype* Y, Dtype* D,
                                  int num,
                                  Dtype* buffer_X2, Dtype* buffer_Y2,
                                  Dtype* buffer_ones, Dtype* buffer_XY,
                                  cudaStream_t stream){
    gk_pow(M, X, Dtype(2), buffer_X2, stream);
    gk_pow(N, Y, Dtype(2), buffer_Y2, stream);
    gk_gemm(CUBLAS_OP_N, CUBLAS_OP_T, M, N, k, Dtype(-2), X, Y,  Dtype(0.), D, stream);
    gk_ger(M, N, buffer_ones, buffer_Y2, Dtype(1), D, stream);

    //TODO: calc shorest distances

    if (num > 1) {
      gk_ger(M, num, buffer_X2, buffer_ones, Dtype(1), D, stream);
    }else{
      gk_axpby(M, Dtype(1), buffer_X2, Dtype(1), D, stream);
    }
  }
}
