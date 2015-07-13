//
// Created by alex on 7/9/15.
//

#include "gkmeans/common.h"
#include "gkmeans/utils/cuda_utils.h"

namespace gkmeans{

  shared_ptr<GKMeans> GKMeans::singleton_;

  GKMeans::GKMeans()
      : cublas_handle_(NULL), curand_generator_(NULL), phase_(GKMeans::SEEDING) {
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS){
      LOG(FATAL)<<"CUBLAS not available, will now halt.";
    }

    if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS){
      LOG(FATAL)<<"CURAND not available, will now halt";
    }

    if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS){
      LOG(FATAL)<<"CUSPARSE not available, will now halt";
    }

    /** Generally, we only need double buffer, so here we create two streams */
    cuda_streams_.resize(2);
    for (size_t i = 0; i < cuda_streams_.size(); ++i){
      CUDA_CHECK(cudaStreamCreate(&cuda_streams_[i]));
    }

    /** Setupa default mat descriptor for cusparse */
    CUSPARSE_CHECK(cusparseCreateMatDescr(&cusparse_descriptor_));

  }
  GKMeans::~GKMeans(){

    /**
     * We should destroy the cuda streams in the de-constructor
     * However, the cuda driver will be deconstructed first, which cause the commented function call to crash.
     * So we comment out this part. But we should be aware that there are two streams to be release.
     */
    for (size_t i = 0; i < cuda_streams_.size(); ++i){
      cudaStreamDestroy(cuda_streams_[i]);
    }

    // destroy cublas handler
    if (cublas_handle_) cublasDestroy(cublas_handle_);

    // destroy curand generator
    if (curand_generator_) curandDestroyGenerator(curand_generator_);

    // destroy cusparse handle
    if (cusparse_handle_) cusparseDestroy(cusparse_handle_);

  };


}