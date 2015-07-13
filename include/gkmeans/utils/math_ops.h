//
// Created by alex on 7/10/15.
//

#ifndef GKMEANS_MATH_OPS_H
#define GKMEANS_MATH_OPS_H

#include "gkmeans/common.h"
#include "gkmeans/utils/cuda_utils.h"

namespace gkmeans {

  /** scale the numbers */
  template <typename Dtype>
  void gk_scale(const int N, const Dtype alpha, Dtype* data, cudaStream_t stream);

  /** axpby operation with CUBLAS*/
  template<typename Dtype>
  void gk_axpby(const int N, const Dtype alpha, const Dtype *X,
                    const Dtype beta, Dtype *Y, cudaStream_t stream);

  /** native atomic add */
  template<typename Dtype>
  void gk_add(const int N, const Dtype *a, const Dtype *b, Dtype *y, cudaStream_t stream);

  /** gemm with CUBLAS */
  template <typename Dtype>
  void gk_gemm(const cublasOperation_t TransA,
                      const cublasOperation_t TransB, const int M, const int N, const int K,
                      const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
                      Dtype* C, cudaStream_t stream);

  /** gemv with CUBLAS */
  template <typename Dtype>
  void gk_gemv(const cublasOperation_t TransA, const int M, const int N,
                      const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
                      Dtype* y, cudaStream_t stream);

  /** power with GPU */
  template<typename Dtype>
  void gk_pow(const int N, const Dtype *X, const Dtype alpha, Dtype *Y, cudaStream_t stream);

  /** ger (outer product) with CUBLAS */
  template<typename Dtype>
  void gk_ger(const int M, const int N, const Dtype *x, const Dtype* y, const Dtype alpha, Dtype *A, cudaStream_t stream);

  /** per row argmax and max with CUDA */
  template <typename Dtype>
  void gk_rmin(const int M, const int N, const Dtype* data, int* max_idx, Dtype* max_val, cudaStream_t stream);

  /** sparse matrix multiplication, used for updating cluster center */
  template <typename Dtype>
  void gk_sparse_gemm2(const cusparseOperation_t TransA, const cusparseOperation_t TransB,
                       const int M, const int N, const int K, const int NNZ,
                       const Dtype alpha, const Dtype* dataA, const int* rowPtrA, const int *colIdxA,
                       const Dtype* B, const Dtype beta, Dtype* C, cudaStream_t stream);

  /** indexed sum **/
  template <typename Dtype>
  void gk_isum(const int M, const int N, const int K, const Dtype* X, const int* DI,
               Dtype* Y, Dtype* ISum, cudaStream_t stream);


  template <typename Dtype>
  void gk_csr2csc(const int M, const int N, const int NNZ,
                  const Dtype* csrData, const int * csrRowPtr, const int* csrColInd,
                  Dtype* cscData, int* cscRowInd, int* cscColPtr, cudaStream_t stream);
}

#endif //GKMEANS_MATH_OPS_H
