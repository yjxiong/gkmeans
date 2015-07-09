//
// Created by alex on 7/9/15.
//

#include "gkmeans/common.h"

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

    /** Generally, we only need double buffer, so here we create two streams */
    for (size_t i = 0; i < 2; ++i){
      cudaStream_t s;
      CUDA_CHECK(cudaStreamCreate(&s));
      cuda_streams_.push_back(s);
    }

  }
  GKMeans::~GKMeans(){
    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (curand_generator_) curandDestroyGenerator(curand_generator_);

    for (size_t i = 0; i < cuda_streams_.size(); ++i){
      CUDA_CHECK(cudaStreamDestroy(cuda_streams_[i]));
    }

  };


}