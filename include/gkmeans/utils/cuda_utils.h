//
// Created by alex on 7/10/15.
//

#ifndef GKMEANS_CUDA_UTILS_H
#define GKMEANS_CUDA_UTILS_H

#include "gkmeans/common.h"

namespace gkmeans {
#define CUDA_CHECK(condition)\
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess)<<" "<<cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition)\
  do {\
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)<<"cublas error :"<<status; \
  } while (0)

#define CUSPARSE_CHECK(condition)\
  do {\
    cusparseStatus_t status = condition; \
    CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS)<<"cusparse error :"<<status; \
  } while (0)

/** Standard number of threads for mainstream GPUs */
  const int CUDA_NUM_THREADS = 512;

/** CUDA: number of blocks for threads. */
  inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  }

// CUDA: grid stride looping
// Since CUDA7, c++11 is supported,
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

}

#define CUBLAS_SET_STREAM(stream) \
  CUBLAS_CHECK(cublasSetStream(GKMeans::cublas_handle(), stream))

#define CUSPARSE_SET_STREAM(stream) \
  CUSPARSE_CHECK(cusparseSetStream(GKMeans::cusparse_handle(), stream))

#endif //GKMEANS_CUDA_UTILS_H
