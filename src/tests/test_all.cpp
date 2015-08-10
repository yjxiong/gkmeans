//
// Created by alex on 7/11/15.
//
#include "gtest/gtest.h"
#include "gkmeans/test_all.h"

int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  google::InstallFailureSignalHandler();
  return RUN_ALL_TESTS();
}