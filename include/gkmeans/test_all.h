//
// Created by alex on 7/11/15.
//

#ifndef GKMEANS_TEST_ALL_H
#define GKMEANS_TEST_ALL_H

#include "gkmeans/common.h"

namespace gkmeans{

  template <typename TypeParam>
  class GKTest : public ::testing::Test {
  public:
    typedef TypeParam Dtype;
  protected:
    GKTest() {
      GKMeans::set_config("device_id", "2");
      gkmeans::GlobalInit();
    }
    virtual ~GKTest() {}
  };

  typedef ::testing::Types<float> TestDtypes;
}

#endif //GKMEANS_TEST_ALL_H
