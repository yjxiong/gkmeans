//
// Created by alex on 7/14/15.
//

#include <chrono>
#include "gtest/gtest.h"
#include "gkmeans/test_all.h"
#include "gkmeans/mat.h"
#include "gkmeans/data_providers.h"

namespace gkmeans {

  template<typename TypeParam>
  class DataProviderTest : public GKTest<TypeParam> {
  public:
    typedef TypeParam Dtype;

    DataProviderTest() { };

    void TestProviderSetup(){

      cudaStream_t t0 = GKMeans::stream(0);

      DummyDataProvider<Dtype> dp(t0);

      Mat<Dtype>* mat = dp.SetUp().get();

      /** Dummy data provider shoud set the data size to (1) */
      EXPECT_EQ((int)mat->count(), 1);

      dp.EndPrefetching();

    }

    void TestDataFlow(){
      cudaStream_t t0 = GKMeans::stream(0);

      DummyDataProvider<Dtype> dp(t0);

      // setup the data provider
      dp.SetUp();

      /* */
      size_t num = 0;

      for (int run = 0; run < 10; ++run) {
        Mat<Dtype> *out_mat = dp.GetData(num).get();

        CHECK_EQ(num, 1);
        CHECK_EQ(out_mat->cpu_data()[0], run);
      }

      dp.EndPrefetching();

    }
  protected:

    virtual void SetUp(){}
    virtual void TearDown(){}
  };

  TYPED_TEST_CASE(DataProviderTest, TestDtypes);

  TYPED_TEST(DataProviderTest, ProviderSetup){
    this->TestProviderSetup();
  }

  TYPED_TEST(DataProviderTest, ProviderStreaming){
    this->TestDataFlow();
  }


  template<typename TypeParam>
  class HDF5DataProviderTest : public GKTest<TypeParam> {
  public:
    typedef TypeParam Dtype;

    HDF5DataProviderTest() { };

    void TestSetup(){
      cudaStream_t t0 = GKMeans::stream(0);

      /** Test config **/
      GKMeans::set_config("data_file", "/media/ssd2/code/GKMeans/data/test_data/data_provider_test.h5");
      GKMeans::set_config("data_name", "data");
      GKMeans::set_config("batch_size", "20");

      HDF5DataProvider<Dtype> dp(t0);

      dp.SetUp();

      size_t num;
      Mat<Dtype> *out_mat = dp.GetData(num).get();
      const Dtype* data = out_mat->cpu_data();
      for (size_t i = 0; i < num; ++i){
        for(size_t j = 0; j < out_mat->shape(1); ++j){
          EXPECT_EQ(data[idx2d(i, j, out_mat->shape(1))], i);
        }
      }

      dp.EndPrefetching();
    }

    void TestDataFlow(){
      cudaStream_t t0 = GKMeans::stream(0);

      /** Test config **/
      GKMeans::set_config("data_file", "/media/ssd2/code/GKMeans/data/test_data/data_provider_test.h5");
      GKMeans::set_config("data_name", "data");
      GKMeans::set_config("batch_size", "20");

      HDF5DataProvider<Dtype> dp(t0);

      dp.SetUp();

      for (int run = 0; run < 20; ++run){
        size_t num;
        Mat<Dtype> *out_mat = dp.GetData(num).get();
        CHECK_EQ(num, 20);
        const Dtype* data = out_mat->cpu_data();
        for (size_t i = 0; i < num; ++i){
          for(size_t j = 0; j < out_mat->shape(1); ++j){
            EXPECT_EQ(data[idx2d(i, j, out_mat->shape(1))], (20 * run)%100 + i);
          }
        }
      }
    }
  protected:
    virtual void SetUp(){};
  };


  TYPED_TEST_CASE(HDF5DataProviderTest, TestDtypes);

  TYPED_TEST(HDF5DataProviderTest, TestSetup){
    this->TestSetup();
  }

  TYPED_TEST(HDF5DataProviderTest, TestDataFlow){
    this->TestDataFlow();
  }
}