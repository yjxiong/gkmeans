// Created by alex on 7/14/15.
//

#include <gkmeans/comp_functions.h>
#include "gkmeans/common.h"
#include "gkmeans/functions.h"
#include "gkmeans/controllers.h"


namespace gkmeans{

  /**
   *  We are manully ensembling the controller here because we know the structure of the model is fixed.
   */
  template<typename Dtype>
  void KMeansController<Dtype>::SetUp(){

    //register controller structure
    FunctionBase<Dtype>* func = new NearestNeighborFunction<Dtype>();
    this->funcs_.push_back(func);
    this->registerFuncName("maximize");

    func = new CenterOfMassFunction<Dtype>();
    this->funcs_.push_back(func);
    this->registerFuncName("estimate");

    /**
     * setup input and output
     */

    /*input*/

    vector<Mat<Dtype>*> input_0, output_0, input_1, output_1;
    vector<int> input_id_0, output_id_0, input_id_1, output_id_1;
    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("X");
    markMat(input_0, input_id_0);
    markMat(input_1, input_id_1);

    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("Y_old");
    input_0.push_back(this->mats_.back());
    markMat(input_0, input_id_0);

    /*intermediate results*/
    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("D");
    markMat(output_0, output_id_0);


    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("DI");
    markMat(output_0, output_id_0);
    markMat(input_1, input_id_1);

    /*output*/
    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("Y_new");
    markMat(output_1, output_id_1);

    this->function_input_vecs_.push_back(input_0);
    this->function_input_vecs_.push_back(input_1);

    this->function_input_id_vecs_.push_back(input_id_0);
    this->function_input_id_vecs_.push_back(input_id_1);

    this->function_output_vecs_.push_back(output_0);
    this->function_output_vecs_.push_back(output_1);

    this->function_output_id_vecs_.push_back(output_id_0);
    this->function_output_id_vecs_.push_back(output_id_1);
    //setup mat shapes

  }


  template<typename Dtype>
  void KMeansController<Dtype>::Step(){

  }

  template<typename Dtype>
  void KMeansController<Dtype>::PostProcess(){

  }

}
