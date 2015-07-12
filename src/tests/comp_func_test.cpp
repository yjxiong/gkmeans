//
// Created by alex on 7/12/15.
//

#include <chrono>
#include "gtest/gtest.h"
#include "gkmeans/test_all.h"
#include "gkmeans/mat.h"
#include "gkmeans/comp_functions.h"

namespace gkmeans {

  template<typename TypeParam>
  class NearestNeighborFunctionTest : public GKTest<TypeParam> {
  public:
    typedef TypeParam Dtype;

    NearestNeighborFunctionTest() { };

    void TestFunctionSetup(){

      NearestNeighborFunction<Dtype> func;

      func.FunctionSetUp(input_vecs_, output_vecs_);

      CHECK_EQ(output_vecs_[0]->shape(0), M);

      CHECK_EQ(output_vecs_[0]->count(), M);
    }

    void TestExecute(){

      cudaStream_t t0 = GKMeans::stream(0);

      NearestNeighborFunction<Dtype> func;
      func.SetUp(input_vecs_, output_vecs_);

      Dtype* x_data = input_vecs_[0]->mutable_cpu_data();
      Dtype* y_data = input_vecs_[1]->mutable_cpu_data();

      for (size_t i = 0; i < M; ++i){
        for (size_t k = 0; k < K; ++k){
          x_data[idx2d(i, k, K)] = i;
        }
      }
      input_vecs_[0]->to_gpu_async(t0);
      for (size_t i = 0; i < N; ++i){
        for (size_t k = 0; k < K; ++k){
          y_data[idx2d(i, k, K)] = i;
        }
      }
      input_vecs_[1]->to_gpu_async(t0);



      func.Execute(input_vecs_, output_vecs_, t0);

      output_vecs_[0]->to_cpu_async(t0);
      output_vecs_[1]->to_cpu_async(t0);

      CUDA_CHECK(cudaStreamSynchronize(t0));
      const Dtype* cpu_result_data = output_vecs_[1]->cpu_data();
      const int* cpu_result_index = (int*)output_vecs_[0]->cpu_data();

      for (size_t i = 0; i < M; ++i){
        if (i < N) {
          EXPECT_EQ(int(i), cpu_result_index[i]);
          EXPECT_NEAR(0, cpu_result_data[i], 1);
        }else{
          EXPECT_EQ(int(N) - 1, cpu_result_index[i]);
          EXPECT_NEAR(cpu_result_data[i], std::pow(int(i - N)  + 1, 2) * K, 10);
        }
      }
    }

  protected:

    virtual void SetUp(){


      //setup data
      M = 1000;
      N = 200;
      K = 50;

      input_vecs_.push_back(new Mat<Dtype>(vector<size_t>({M, K})));
      input_vecs_.push_back(new Mat<Dtype>(vector<size_t>({N, K})));

      output_vecs_.push_back(new Mat<Dtype>());
      output_vecs_.push_back(new Mat<Dtype>());
    }

    virtual void TearDown(){
      for (size_t i = 0; i < input_vecs_.size(); ++i){
        delete input_vecs_[i];
      }
      input_vecs_.clear();
      for (size_t i = 0; i < output_vecs_.size(); ++i){
        delete output_vecs_[i];
      }
      output_vecs_.clear();
    }


    vector<Mat<Dtype>* > input_vecs_;
    vector<Mat<Dtype>* > output_vecs_;

    size_t M, N, K;
  };


  TYPED_TEST_CASE(NearestNeighborFunctionTest, TestDtypes);

  TYPED_TEST(NearestNeighborFunctionTest, SetUpTest){
    this->TestFunctionSetup();
  }

  TYPED_TEST(NearestNeighborFunctionTest, ExecuteTest){
    this->TestExecute();
  }
}