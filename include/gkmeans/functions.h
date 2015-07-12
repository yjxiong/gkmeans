//
// Created by alex on 7/12/15.
//

#ifndef GKMEANS_FUNCTIONS_H
#define GKMEANS_FUNCTIONS_H

#include "gkmeans/common.h"
#include "gkmeans/mat.h"

namespace gkmeans{

  /**
   * @brief the base class of all functions in the solver
   * This class defines the functions which every function class should implement
   */
  template <typename Dtype>
  class FunctionBase{
  public:

    //not to be overridden
    explicit FunctionBase() {
    }

    virtual ~FunctionBase(){};

    void SetUp(const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec){
      FunctionSetUp(input_mat_vec, output_mat_vec);
    }

    /**
     * Number of input
     */
    virtual const int NumInputs() = 0;

    /**
     * Number of output
     */
    virtual const int NumOutputs() = 0;

    /**
     * Set up the function body
     */
    virtual void FunctionSetUp(const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec) = 0;

    /**
     * Returns the name of the function
     */
    inline virtual const char* FunctionType() = 0;


    /**
     * Set up the function body
     */
    virtual void Execute(const vector<Mat<Dtype> *>& input_mat_vec, const vector<Mat<Dtype> *>& output_mat_vec, cudaStream_t stream) = 0;
  };

}

#endif //GKMEANS_FUNCTIONS_H
