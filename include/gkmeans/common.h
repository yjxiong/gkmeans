//
// Created by alex on 7/9/15.
//

#ifndef GKMEANS_COMMON_H

//general include
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <set>
#include <utility>
#include <algorithm>

//include CUDA libraries
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>

//glog for logging
#include <glog/logging.h>

//shared_ptr
#include <memory>

using std::shared_ptr;
using std::pair;
using std::vector;
using std::printf;
using std::cout;
using std::ifstream;
using std::ofstream;
using std::map;
using std::make_pair;
using std::set;

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


namespace gkmeans {
  //common code here
  class GKMeans{

  public:
    ~GKMeans();
    inline static GKMeans& Get(){
      if (!singleton_){
        singleton_.reset(new GKMeans);
      }
      return *singleton_;
    }

    enum Phase {
      SEEDING,
      CLUSTERING
    };

    //getter-setters
    inline static Phase phase(){return Get().phase_;}
    inline static void set_phase(Phase new_phase){Get().phase_ = new_phase;}

    inline cudaStream_t stream(int i){return Get().cuda_streams_[i];}
    inline const vector<cudaStream_t>& stream_vec(){return Get().cuda_streams_;}

  protected:
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_generator_;

    vector<cudaStream_t> cuda_streams_;

    Phase phase_;

    static shared_ptr<GKMeans> singleton_;


  private:
    //disable copy and sign constructor
    GKMeans(const GKMeans&);
    GKMeans& operator=(const GKMeans&);

    GKMeans();
  };
}


#define GKMEANS_COMMON_H

#endif //GKMEANS_COMMON_H
