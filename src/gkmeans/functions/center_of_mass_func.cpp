//
// Created by alex on 7/12/15.
//

#include "gkmeans/common.h"
#include "gkmeans/functions.h"
#include "gkmeans/comp_functions.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/distance_metrics.h"
#include "gkmeans/mat.h"

namespace gkmeans{

  template<typename Dtype>
  void CenterOfMassFunction<Dtype>::FunctionSetUp(
      const vector<Mat<Dtype> *> &input_mat_vec, const vector<Mat<Dtype> *> &output_mat_vec) {

  }

  template<typename Dtype>
  void CenterOfMassFunction<Dtype>::Execute(
      const vector<Mat<Dtype> *> &input_mat_vec, const vector<Mat<Dtype> *> &output_mat_vec,
      cudaStream_t stream) {

  }

  INSTANTIATE_CLASS(CenterOfMassFunction);

}