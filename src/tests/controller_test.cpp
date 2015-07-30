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

    void TestKMeansSeed() {
      KMeansController<Dtype> controller;
      controller.SetUp();

      controller.Seed();

      Mat<Dtype>* mat = controller.mats()[1];

      const Dtype* y_data = mat->cpu_data();
      for (int i = 0; i < 10; ++i){
        const Dtype* data = controller.data_providers()[0]->DirectAccess(pseudo_random_seq_[i]);
        for (int col = 0; col < 10; ++col){
          EXPECT_EQ(data[col], y_data[col + i * 20]);
        }
      }
    }

    void TestKmeansSolve() {
      KMeansController<Dtype> controller;
      controller.SetUp();

      controller.Solve(1); //do one iteration

      Mat<Dtype>* center_mat = controller.mats()[1];

      const Dtype* center_data = center_mat->cpu_data();

      float results[] = {16.5000000000000,
          90,
          26.5000000000000,
          77,
          7,
          71,
          65,
          20.50000,
          37.5000000000000,
          53};

      EXPECT_NEAR(center_data[0], results[0], 0.1);
      EXPECT_NEAR(center_data[1 * 20], results[1], 0.1);
      EXPECT_NEAR(center_data[4 * 20], results[4], 0.1);

      controller.data_providers()[0]->EndPrefetching();

    }

  protected:

    void SetUp(){
      /** Test config **/
      GKMeans::set_config("data_file", "/media/ssd2/code/GKMeans/data/test_data/data_provider_test.h5");
      GKMeans::set_config("data_name", "data");
      GKMeans::set_config("batch_size", "20");
      GKMeans::set_config("n_cluster", "10");
      GKMeans::set_config("random_seed", "1");

      pseudo_random_seq_.resize(100);
      std::iota(pseudo_random_seq_.begin(), pseudo_random_seq_.end(), 0);

      std::default_random_engine engine;
      engine.seed(1);
      std::shuffle(pseudo_random_seq_.begin(), pseudo_random_seq_.end(), engine);
    }

    void TearDown(){

    }

    vector<int> pseudo_random_seq_;

  };

  TYPED_TEST_CASE(KMeansControllerTest, TestDtypes);

  TYPED_TEST(KMeansControllerTest, TestSetup){
    this->TestKmeansSetup();
  }

  TYPED_TEST(KMeansControllerTest, TestSeed){
    this->TestKMeansSeed();
  }

  TYPED_TEST(KMeansControllerTest, TestRun){
    this->TestKmeansSolve();
  }
}