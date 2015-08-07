//
// Created by alex on 8/6/15.
//

#ifndef GKMEANS_IO_H
#define GKMEANS_IO_H

#include "gkmeans/common.h"
#include "gkmeans/mat.h"

namespace gkmeans {

  template <typename Dtype>
  int WriteDataToHDF5(string file_name, string mat_name, Mat<Dtype>* mat);

  template <typename Dtype>
  int LoadDataFromHDF5(string file_name, string mat_name, Mat<Dtype>* mat);

}

#endif //GKMEANS_IO_H
