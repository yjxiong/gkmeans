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
  }
  GKMeans::~GKMeans(){
    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (curand_generator_) curandDestroyGenerator(curand_generator_);
  };


}