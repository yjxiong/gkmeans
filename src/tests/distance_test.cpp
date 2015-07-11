//
// Created by alex on 7/12/15.
//

#include <chrono>
#include "gtest/gtest.h"
#include "gkmeans/test_all.h"
#include "gkmeans/mat.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/distance_metrics.h"

namespace gkmeans {

  template<typename TypeParam>
  class DistanceTest : public GKTest<TypeParam> {
  public:
    typedef TypeParam Dtype;

    size_t M, N, K;

    DistanceTest() { };

    void TestEucileanDistance(){
      size_t max_num = std::max(M, N);
      size_t max_dim = std::max(max_num, K);
      Mat<Dtype> buffer_x2(vector<size_t >({M, K}));
      Mat<Dtype> buffer_Y2(vector<size_t >({N, K}));
      Mat<Dtype> buffer_ones(vector<size_t>({max_dim, 1}));
      Mat<Dtype> buffer_norm(vector<size_t>({max_num}));
      M_result.reset(new Mat<Dtype>({M, N}));

      cudaStream_t t0 = GKMeans::stream(0);


      Dtype* cpu_data = buffer_ones.mutable_cpu_data();
      for (size_t i = 0; i < buffer_ones.count(); ++i){
        cpu_data[i] = 1;
      }
      buffer_ones.to_gpu_async(t0);
      cpu_data = M_left->mutable_cpu_data();
      for (size_t i = 0; i < M_left->shape(0); ++i){
        for (size_t k = 0; k < M_left->shape(1); ++k){
          cpu_data[idx2d(i, k, M_left->shape(1))] = 1;
        }
      }

      M_left->to_gpu_async(t0);
      cpu_data = M_right->mutable_cpu_data();
      for (size_t i = 0; i < M_right->shape(0); ++i){
        for (size_t k = 0; k < M_right->shape(1); ++k){
          cpu_data[idx2d(i, k, M_right->shape(1))] = i;
        }
      }
      M_right->to_gpu_async(t0);



      Dtype* result_data = M_result->mutable_gpu_data();
      Dtype* x2_data = buffer_x2.mutable_gpu_data();
      Dtype* y2_data = buffer_Y2.mutable_gpu_data();
      const Dtype* ones_data = buffer_ones.gpu_data();
      Dtype* norm_data = buffer_norm.mutable_gpu_data();



      const Dtype* left_data = M_left->gpu_data();
      const Dtype* right_data = M_right->gpu_data();

      CUDA_CHECK(cudaStreamSynchronize(t0));
      auto start = std::chrono::system_clock::now();

      gk_euclidean_dist<Dtype>(M, N, K, left_data, right_data, result_data,
                      x2_data, y2_data, ones_data, norm_data,
                               t0);

      CUDA_CHECK(cudaStreamSynchronize(t0));
      cout<<"Total distance time: "
        <<std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - start).count()/1000<<" us\n";

      M_result->to_cpu_async(t0);

      const Dtype* rst_data = M_result->cpu_data();
      for (size_t i = 0; i < M_result->shape(0); ++i){
        for (size_t j = 0; j < M_result->shape(1); ++j)
          EXPECT_NEAR(rst_data[idx2d(i, j, N)], std::pow(float(j) - 1, 2) * K, .1);
      }

    }
  protected:

    shared_ptr<Mat<Dtype> > M_left, M_right;
    shared_ptr<Mat<Dtype> > M_result;

    virtual void SetUp(){

      M = 100;
      N = 30;
      K = 20;
      M_left.reset(new Mat<Dtype>(vector<size_t >({M, K})));
      M_right.reset(new Mat<Dtype>(vector<size_t >({N, K})));
    }

    virtual void TearDown() {
      /*M_left.reset();
      M_right.reset();
      M_result.reset();*/
    }
  };

  TYPED_TEST_CASE(DistanceTest, TestDtypes);

  TYPED_TEST(DistanceTest, EuclideanDistance){
    this->TestEucileanDistance();
  }
}