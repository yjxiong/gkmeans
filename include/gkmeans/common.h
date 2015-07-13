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

//thrust
#include <thrust/device_vector.h>

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

namespace gkmeans {
  //common code here
  class GKMeans{

  public:
    ~GKMeans();
    inline static GKMeans& Get(){
      if (!singleton_){
        singleton_.reset(new GKMeans());
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

    inline static cublasHandle_t cublas_handle(){return Get().cublas_handle_;}
    inline static curandGenerator_t curand_generator(){return Get().curand_generator_;}
    inline static cusparseHandle_t cusparse_handle(){return Get().cusparse_handle_;}
    inline static cudaStream_t stream(int i){return Get().cuda_streams_[i];}
    inline static const vector<cudaStream_t>& stream_vec(){return Get().cuda_streams_;}
    inline static cusparseMatDescr_t cusparse_descriptor(){return Get().cusparse_descriptor_;}

  protected:
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_generator_;
    cusparseHandle_t cusparse_handle_;
    vector<cudaStream_t> cuda_streams_;
    cusparseMatDescr_t cusparse_descriptor_;

    Phase phase_;

    static shared_ptr<GKMeans> singleton_;


  private:
    //disable copy and sign constructor
    GKMeans(const GKMeans&);
    GKMeans& operator=(const GKMeans&);

    GKMeans();
  };
}

//row-major based 1d access
#define idx2d(row, col, ldx) \
  row * ldx + col

#define idx3d(x1, x2, x3, dim1, dim2) \
  (x1 * dim1 + x2) * dim2 + x3


#define INSTANTIATE_CLASS(classname) \
  char gChar##classname; \
  template class classname<float>

#define GKMEANS_COMMON_H

#endif //GKMEANS_COMMON_H
