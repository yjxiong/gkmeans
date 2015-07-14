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

      Mat<Dtype>* mat = dp.SetUp();

      EXPECT_EQ((int)mat->count(), 1);

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
}