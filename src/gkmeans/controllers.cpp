// Created by alex on 7/14/15.
//

#include <gkmeans/comp_functions.h>
#include "gkmeans/common.h"
#include "gkmeans/functions.h"
#include "gkmeans/controllers.h"
#include "gkmeans/data_providers.h"


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

    /** Build data provider **/
    DataProviderBase<Dtype>* dp = new HDF5DataProvider<Dtype>(GKMeans::stream(1));
    this->data_providers_.push_back(dp);
    Mat<Dtype>* source_mat = dp->SetUp();

    /**
     * setup input and output
     */

    /*input*/

    vector<Mat<Dtype>*> input_0, output_0, input_1, output_1;
    vector<int> input_id_0, output_id_0, input_id_1, output_id_1;

    /** The data provider will build the first input mat */
    this->mats_.push_back(source_mat);
    this->registerMatName("X");
    this->markMat(input_0, input_id_0);
    this->markMat(input_1, input_id_1);

    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("Y_old");
    input_0.push_back(this->mats_.back());
    this->markMat(input_0, input_id_0);

    /*intermediate results*/
    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("DI");
    this->markMat(output_0, output_id_0);
    this->markMat(input_1, input_id_1);


    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("D");
    this->markMat(output_0, output_id_0);
    this->markMat(input_1, input_id_1);

    /*output*/
    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("Y_new");
    this->markMat(output_1, output_id_1);

    this->mats_.push_back(new Mat<Dtype>());
    this->registerMatName("Isum");
    this->markMat(output_1, output_id_1);

    this->function_input_vecs_.push_back(input_0);
    this->function_input_vecs_.push_back(input_1);

    this->function_input_id_vecs_.push_back(input_id_0);
    this->function_input_id_vecs_.push_back(input_id_1);

    this->function_output_vecs_.push_back(output_0);
    this->function_output_vecs_.push_back(output_1);

    this->function_output_id_vecs_.push_back(output_id_0);
    this->function_output_id_vecs_.push_back(output_id_1);

    //setup mat shapes
    M_ = this->data_providers_[0]->round_size();
    K_ = this->mats_[0]->shape(1);
    N_ = std::stoul(GKMeans::get_config("n_cluster"));

    this->mats_[1]->Reshape(vector<size_t>({N_, K_}));

    for (size_t i = 0; i < this->funcs_.size(); ++i){
      this->funcs_[i]->SetUp(this->function_input_vecs_[i], this->function_output_vecs_[i]);
    }

  }

  template<typename Dtype>
  void KMeansController<Dtype>::Seed(){
    string seed_type = GKMeans::get_config("seed_type");

    if ((seed_type == "random") ||(seed_type == "")){
      // use random seeding
      vector<size_t> src;
      src.resize(N_);
      std::iota(src.begin(), src.end(), 0);
      std::default_random_engine engine;
      std::shuffle(src.begin(), src.end(), engine);

      Dtype* y_data = this->mats_[1]->mutable_cpu_data();
      for(size_t i = 0; i < N_; ++i){
        Dtype* row_data = this->data_providers_[0]->DirectAccess(i);
        std::memcpy(y_data, row_data, K_ * sizeof(Dtype));
        y_data += K_;
      }
    }
  }


  template<typename Dtype>
  void KMeansController<Dtype>::Step(){

  }

  template<typename Dtype>
  void KMeansController<Dtype>::PostProcess(){

  }

  INSTANTIATE_CLASS(KMeansController);

}
