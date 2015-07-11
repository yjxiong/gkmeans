//
// Created by alex on 7/11/15.
//

#include <chrono>
#include "gtest/gtest.h"
#include "gkmeans/test_all.h"
#include "gkmeans/mem.h"
#include "gkmeans/mat.h"
#include "gkmeans/utils/math_ops.h"

namespace gkmeans {

  template<typename TypeParam>
  class MemTest : public GKTest<TypeParam> {
  public:
    typedef TypeParam Dtype;
    MemTest(){};

    void TestBasicAllocation(){
      mem_.reset(new Mem(20));

      const void *cpu_data = mem_->cpu_data();
      EXPECT_TRUE(cpu_data != NULL);

      void *mcpu_data = mem_->mutable_cpu_data();
      EXPECT_TRUE(mcpu_data != NULL);

      (static_cast<float*> (mcpu_data))[0] = 1;

      EXPECT_EQ(1, (static_cast<float*> (mcpu_data))[0]);

      const void* gpu_data = mem_->gpu_data();
      EXPECT_TRUE(gpu_data != NULL);

      cpu_data = mem_->cpu_data();
      EXPECT_EQ(1, (static_cast<const float*> (cpu_data))[0]);

      void* mgpu_data = mem_->mutable_gpu_data();
      EXPECT_TRUE(mgpu_data != NULL);

      cpu_data = mem_->cpu_data();
      EXPECT_EQ(1, (static_cast<float*> (mcpu_data))[0]);

      mem_.reset();
    }
  protected:

    shared_ptr<Mem> mem_;

    virtual void SetUp(){
      cudaSetDevice(0);
    }

    virtual void TearDown() {}

  };

  TYPED_TEST_CASE(MemTest, TestDtypes);

  TYPED_TEST(MemTest, Allocation){
    this->TestBasicAllocation();
  }


  template<typename TypeParam>
  class MatTest : public GKTest<TypeParam> {
  public:
    typedef TypeParam Dtype;
    MatTest() { };

    void TestBasicAllocation(){
      mat_.reset(new Mat<Dtype>(vector<size_t>({2,3,4})));

      EXPECT_EQ((size_t)24, mat_->count());

      const Dtype* cpu_data = mat_->cpu_data();
      EXPECT_TRUE(cpu_data != NULL);

      Dtype * mcpu_data = mat_->mutable_cpu_data();
      EXPECT_TRUE(mcpu_data != NULL);
      mcpu_data[0] = 2;

      const Dtype * gpu_data = mat_->gpu_data();
      EXPECT_TRUE(gpu_data != NULL);

      Dtype* mgpu_data = mat_->mutable_gpu_data();
      EXPECT_TRUE(mgpu_data != NULL);

      gk_add<Dtype>(20, mgpu_data, mgpu_data, mgpu_data, 0);

      cpu_data = mat_->cpu_data();
      EXPECT_EQ(4, cpu_data[0]);

      mat_.reset();

    }

  void TestAsyncOperation(){
    mat_.reset(new Mat<Dtype>(vector<size_t>({100, 224, 224})));
    EXPECT_EQ((size_t)100 * 224 * 224, mat_->count());

    Mat<Dtype> m_buf(vector<size_t>({100, 224, 224}));

    cudaStream_t t0, t1;
    CUDA_CHECK(cudaStreamCreate(&t0));
    CUDA_CHECK(cudaStreamCreate(&t1));

    Dtype* mcpu_data = mat_->mutable_cpu_data();

    for (size_t i = 0; i < mat_->count(); ++i){
      mcpu_data[i] = i;
    }

    mat_->to_gpu_async(t0);

    mcpu_data = m_buf.mutable_cpu_data();

    for (size_t i = 0; i < mat_->count(); ++i){
      mcpu_data[i] = 3 * i;
    }

    m_buf.to_gpu_async(t1);

    auto start = std::chrono::system_clock::now();

    Dtype* mgpu_data = mat_->mutable_gpu_data();

    gk_add(mat_->count(), mgpu_data, mgpu_data, mgpu_data, t0);

    mat_->to_cpu_async(t0);

    mgpu_data = m_buf.mutable_gpu_data();

    gk_add(m_buf.count(), mgpu_data, mgpu_data, mgpu_data, t1);

    m_buf.to_cpu_async(t1);

    const Dtype* cpu_data;
    cpu_data = mat_->cpu_data();
    auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);


    cout<<"Time "<<elapse.count()<<" ms\n";
    for (size_t i = 0; i < mat_->count(); ++i){
      EXPECT_EQ(cpu_data[i],  2 * i);
    }

     cpu_data = m_buf.cpu_data();

    for (size_t i = 0; i < mat_->count(); ++i){
      EXPECT_EQ(cpu_data[i],  6 * i);
    }

    mat_.reset();
  }
  protected:

    shared_ptr<Mat<Dtype>> mat_;

    virtual void SetUp(){
      cudaSetDevice(0);
    }

    virtual void TearDown() {}
  };

  TYPED_TEST_CASE(MatTest, TestDtypes);

  TYPED_TEST(MatTest, Allocation){
    this->TestBasicAllocation();
  }

  TYPED_TEST(MatTest, AsyncOperation){
    this->TestAsyncOperation();
  }
}