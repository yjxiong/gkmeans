//
// Created by alex on 7/12/15.
//

#ifndef GKMEANS_COMP_FUNCTIONS_H
#define GKMEANS_COMP_FUNCTIONS_H

#include "gkmeans/common.h"
#include "gkmeans/functions.h"

namespace gkmeans{

  /**
   * @Calculate the shortest pair-wise distance between two matrix
   * The two matrix X and Y are in row-major format
   */
  template <typename Dtype>
  class NearestNeighborFunction : public FunctionBase<Dtype> {
  public:

    explicit NearestNeighborFunction():FunctionBase<Dtype>(){};

    /**
     * @brief inputs 2 mats
     * 1. X
     * 2. Y
     */
    inline virtual const int NumInputs(){return 2;}

    /**
     * @brief outputs 2 mats
     * del 1. Assignment matrix DI
     * 2. Distance matrix D
     */
    inline virtual const int NumOutputs(){return 2;}

    /**
     * @brief this function is called "NearestNeighbor"
     */
    inline virtual const char* FunctionType(){return "NereastNeighbor";}

    virtual void FunctionSetUp(const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec);

    virtual void Execute(const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec, cudaStream_t stream);

  protected:

    shared_ptr<Mat<Dtype>> buffer_X2_, buffer_Y2_, buffer_XY_, buffer_ones_, buffer_norm_;
    size_t m_, n_, k_;
    size_t max_num_, max_dim_;

  };

  /**
   * @Calculate the center of mass for each cluster
   */
  template <typename Dtype>
  class CenterOfMassFunction: public FunctionBase<Dtype>{
  public:

    /**
     * @brief inputs 2 mats
     * 1. X
     * 2. Assignment matrix, DI
     */
    inline virtual const int NumInputs(){return 2;}

    /**
     * @brief outputs 1 mats
     * 1. Accumulated cluster centers, Y'
     */
    inline virtual const int NumOutputs(){return 1;}

    inline virtual const char* FunctionType(){return "CenterOfMass";}

    virtual void FunctionSetUp(const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec);

    virtual void Execute(const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec, cudaStream_t stream);

  protected:
    shared_ptr<Mat<Dtype>> buffer_temp_center_;
  };
}

#endif //GKMEANS_COMP_FUNCTIONS_H
