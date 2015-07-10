//
// Created by alex on 7/10/15.
//

#ifndef GKMEANS_DISTANCE_METRICS_H
#define GKMEANS_DISTANCE_METRICS_H

#include "gkmeans/common.h"

namespace gkmeans {
  template<typename Dtype>
  void gk_euclidean_dist(int M, int N, int k, const Dtype *X, const Dtype *Y, Dtype *D,
                         Dtype *buffer_X2, Dtype* buffer_Y2, Dtype* buffer_ones,
                         cudaStream_t stream);

  template<typename Dtype>
  void gk_shortest_euclidean_dist(int M, int N, int k, const Dtype *X, const Dtype *Y, Dtype *D,
                                  int num,
                                  Dtype *buffer_X2, Dtype* buffer_Y2, Dtype* buffer_ones, Dtype* buffer_XY,
                                  cudaStream_t stream);
}
#endif //GKMEANS_DISTANCE_METRICS_H
