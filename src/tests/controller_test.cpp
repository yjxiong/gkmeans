//
// Created by alex on 7/16/15.
//

#include <chrono>
#include "gtest/gtest.h"
#include "gkmeans/test_all.h"
#include "gkmeans/mat.h"
#include "gkmeans/data_providers.h"
#include "gkmeans/controllers.h"

namespace gkmeans {

  template<typename TypeParam>
  class KMeansControllerTest : public GKTest<TypeParam> {
  public:
    typedef TypeParam Dtype;

    KMeansControllerTest() { };

    void TestKmeansSetup(){
      KMeansController<Dtype> controller;
      controller.SetUp();

//      vector<FunctionBase<Dtype>* >& funcs = controller.funcs();
      vector<Mat<Dtype>* >& mats = controller.mats();

      map<string, int>& func_name_mapping = controller.name_func_indices();

      /** check function names */
      EXPECT_EQ(func_name_mapping.at("maximize"), 0);
      EXPECT_EQ(func_name_mapping.at("estimate"), 1);

      /** check mat links */
      Mat<Dtype>* src_mat = mats[0];
      EXPECT_EQ(controller.mat_names()[0], "X");
      EXPECT_EQ(src_mat->shape().size(), 2);
      EXPECT_EQ(src_mat->shape(0), 20);
      EXPECT_EQ(src_mat->shape(1), 20);

      Mat<Dtype>* y_old = mats[1];
      EXPECT_EQ(controller.mat_names()[1], "Y_old");
      EXPECT_EQ(y_old->shape().size(), 2);
      EXPECT_EQ(y_old->shape(0), 10);
      EXPECT_EQ(y_old->shape(1), 20);

      Mat<Dtype>* D = mats[2];
      EXPECT_EQ(controller.mat_names()[2], "DI");
      EXPECT_EQ(D->shape().size(), 1);
      EXPECT_EQ(D->shape(0), 20);

      Mat<Dtype>* DI = mats[3];
      EXPECT_EQ(controller.mat_names()[3], "D");
      EXPECT_EQ(DI->shape().size(), 1);
      EXPECT_EQ(DI->shape(0), 20);

      Mat<Dtype>* Y_new = mats[4];
      EXPECT_EQ(controller.mat_names()[4], "Y_new");
      EXPECT_EQ(Y_new->shape().size(), 2);
      EXPECT_EQ(Y_new->shape(0), 10);
      EXPECT_EQ(Y_new->shape(1), 20);

      Mat<Dtype>* Isum = mats[4];
      EXPECT_EQ(controller.mat_names()[5], "Isum");
      EXPECT_EQ(Isum->shape().size(), 2);
      EXPECT_EQ(Isum->shape(0), 10);
      EXPECT_EQ(Isum->shape(1), 20);

    }

  protected:

    void SetUp(){
      /** Test config **/
      GKMeans::set_config("data_file", "/media/ssd2/code/GKMeans/data/test_data/data_provider_test.h5");
      GKMeans::set_config("data_name", "data");
      GKMeans::set_config("batch_size", "20");
      GKMeans::set_config("n_cluster", "10");
    }

    void TearDown(){

    }

    size_t M_ = 10;
  };

  TYPED_TEST_CASE(KMeansControllerTest, TestDtypes);

  TYPED_TEST(KMeansControllerTest, TestSetup){
    this->TestKmeansSetup();
  }

  TYPED_TEST(KMeansControllerTest, TestSeed){

  }

  TYPED_TEST(KMeansControllerTest, TestRun){

  }
}