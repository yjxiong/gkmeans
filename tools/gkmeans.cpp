//
// Created by alex on 7/9/15.
//

#include "gkmeans/common.h"

using gkmeans::GKMeans;

int main(int argc, char** argv){

  FLAGS_logtostderr = true;

  google::InitGoogleLogging(argv[0]);

  LOG(INFO)<<"Now phase: "<<GKMeans::phase();
}