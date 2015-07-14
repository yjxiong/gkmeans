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

  /**
   * sparse matrix multiplication, used for updating cluster center
   * WARNING: Output of this function is a column-major matrix.
   * */
  template <>
  void gk_sparse_gemm2<float>(const cusparseOperation_t TransA, const cusparseOperation_t TransB,
                       const int M, const int N, const int K, const int NNZ,
                       const float alpha, const float* dataA, const int* rowPtrA, const int *colIdxA,
                       const float* B, const float beta, float* C, cudaStream_t stream){

    //since our dense matrix is in row-major format, we need to reverse the tranpose paramter here
    cusparseOperation_t transB = (TransB == CUSPARSE_OPERATION_NON_TRANSPOSE)?
                                 CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transA = TransA;

    if ((transA==CUSPARSE_OPERATION_TRANSPOSE) && (transB==CUSPARSE_OPERATION_TRANSPOSE)){
      LOG(FATAL)<<"Trans(A)*Trans(B) is not supported by cusparse, will now halt";
    }

    int ldb = (TransB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? N : K;
    int ldc = (TransA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? M : K;
    CUSPARSE_CHECK(cusparseSetStream(GKMeans::cusparse_handle(), stream));
    CUSPARSE_CHECK(cusparseScsrmm2(GKMeans::cusparse_handle(), transA, transB, M, N, K, NNZ, &alpha,
                                   GKMeans::cusparse_descriptor(), dataA, rowPtrA, colIdxA,
                                   B, ldb, &beta, C, ldc));

  }

  /**
   * Transpose a CSC sparse matrix
   */
  template <>
  void gk_csr2csc<float>(const int M, const int N, const int NNZ,
                  const float* csrData, const int * csrRowPtr, const int* csrColInd,
                         float* cscData, int* cscRowInd, int* cscColPtr, cudaStream_t stream){
    CUSPARSE_CHECK(cusparseSetStream(GKMeans::cusparse_handle(), stream));
    CUSPARSE_CHECK(cusparseScsr2csc(GKMeans::cusparse_handle(), M, N, NNZ,
                   csrData, csrRowPtr, csrColInd,
                   cscData, cscRowInd, cscColPtr,
                   CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO));
  }

  template  void gk_gpu_set<float>(const size_t Count, float* data, int val, cudaStream_t stream);

}
