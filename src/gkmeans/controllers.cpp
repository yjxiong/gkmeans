// Created by alex on 7/14/15.
//

#include <gkmeans/comp_functions.h>
#include "gkmeans/common.h"
#include "gkmeans/functions.h"
#include "gkmeans/controllers.h"
#include "gkmeans/data_providers.h"
#include "gkmeans/utils/math_ops.h"
#include "gkmeans/utils/io.h"


namespace gkmeans{

  /**
   *  We are manully ensembling the controller here because we know the structure of the model is fixed.
   */
  template<typename Dtype>
  void KMeansController<Dtype>::SetUp(){


    //register controller structure
    FunctionBase<Dtype>* func = new NearestNeighborFunction<Dtype>();
    this->funcs_.push_back(shared_ptr<FunctionBase<Dtype>>(func));
    this->registerFuncName("maximize");

    func = new CenterOfMassFunction<Dtype>();
    this->funcs_.push_back(shared_ptr<FunctionBase<Dtype>>(func));
    this->registerFuncName("estimate");

    /** Build data provider **/
    shared_ptr<DataProviderBase<Dtype> > dp( new HDF5DataProvider<Dtype>(GKMeans::stream(1)));
    this->data_providers_.push_back(dp);
    shared_ptr<Mat<Dtype> > source_mat = dp->SetUp();

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
    batch_size_ = source_mat->shape(0);

    this->mats_.push_back(shared_ptr<Mat<Dtype>>(new Mat<Dtype>()));
    this->registerMatName("Y_old");
    this->markMat(input_0, input_id_0);

    /*intermediate results*/
    this->mats_.push_back(shared_ptr<Mat<Dtype>>(new Mat<Dtype>()));
    this->registerMatName("DI");
    this->markMat(output_0, output_id_0);
    this->markMat(input_1, input_id_1);


    this->mats_.push_back(shared_ptr<Mat<Dtype>>(new Mat<Dtype>()));
    this->registerMatName("D");
    this->markMat(output_0, output_id_0);

    this->mats_.push_back(shared_ptr<Mat<Dtype>>(new Mat<Dtype>()));
    this->registerMatName("Y_new");
    this->markMat(output_1, output_id_1);

    this->mats_.push_back(shared_ptr<Mat<Dtype>>(new Mat<Dtype>()));
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
    M_ = dp->round_size();
    K_ = this->mats_[0]->shape(1);
    N_ = std::stoul(GKMeans::get_config("n_cluster"));

    this->mats_[1]->Reshape(vector<size_t>({N_, K_}));

    for (size_t i = 0; i < this->funcs_.size(); i++){
      this->funcs_[i]->SetUp(this->function_input_vecs_[i], this->function_output_vecs_[i]);
    }

    //setup output mats
    this->int_outputs_.push_back(
        shared_ptr<Mat<int> >(new Mat<int>(vector<size_t>({M_})))
    );
    this->numeric_outputs_.push_back(shared_ptr<Mat<Dtype> >(
        new Mat<Dtype>(vector<size_t>({N_, K_})))
    );

  }

  template<typename Dtype>
  void KMeansController<Dtype>::Seed(){
    string seeding_type = GKMeans::get_config("seeding_type");
    string random_seed_string = GKMeans::get_config("random_seed");

    if (seeding_type == "random" || seeding_type == ""){
      // use random seeding
      vector<size_t> src;
      src.resize(M_);
      std::iota(src.begin(), src.end(), 0);
      std::default_random_engine engine;
      if (random_seed_string != ""){
        engine.seed(std::stoul(random_seed_string));
      }else{
        engine.seed(std::random_device{}());
      }
      std::shuffle(src.begin(), src.end(), engine);

      Dtype* y_data = this->mats_[1]->mutable_cpu_data();
      for(size_t i = 0; i < N_; ++i){
        Dtype* row_data = this->data_providers_[0]->DirectAccess(src[i]);
        std::memcpy(y_data, row_data, K_ * sizeof(Dtype));
        y_data += K_;
      }
    }else if (seeding_type == "precomputed"){
      //load precomputed seeds stored in a data file.
      LoadDataFromHDF5<float>(
          GKMeans::get_config("precomputed_seed_file"), GKMeans::get_config("precomputed_seed_name"),
          this->mats_[1].get()
      );

    }else {
      LOG(FATAL)<<"Seeding type \""<<seeding_type<<"\" not supported";
    }
}

  template<typename Dtype>
  void KMeansController<Dtype>::Step(){
    /** one iteration includes maximization and estimation**/
    bool iter_finised = false;
    gk_gpu_set(this->mats_[4]->count(), this->mats_[4]->mutable_gpu_data(), 0, GKMeans::stream(0));
    gk_gpu_set(this->mats_[5]->count(), this->mats_[5]->mutable_gpu_data(), 0, GKMeans::stream(0));
    while (!iter_finised){
      size_t batch_num = 0;
      this->mats_[0] = this->data_providers_[0]->GetData(batch_num);
//      CHECK(batch_num, batch_size_);

      if ( this->data_providers_[0]->current_index()  == 0){
        iter_finised = true;
      }
      //swapping in the data buffer for this batch
      this->function_input_vecs_[0][0] = this->mats_[0].get();
      this->function_input_vecs_[1][0] = this->mats_[0].get();

      if (batch_num < batch_size_) {
        // turn on trailing mode if reaching the final part of the data
        LOG(INFO)<<"Entering traling mode with "<< batch_num <<" samples";

        for (size_t i = 0; i < this->funcs_.size(); ++i){
          // run functions
          this->funcs_[i]->SetTrailingMode(batch_num);

          this->funcs_[i]->Execute(this->function_input_vecs_[i], this->function_output_vecs_[i], GKMeans::stream(0));

          //turn off trailing mode after finish
          this->funcs_[i]->UnsetTrailingMode();
        }
      }else {
        //run normal execution
        for (size_t i = 0; i < this->funcs_.size(); ++i) {
          this->funcs_[i]->Execute(this->function_input_vecs_[i], this->function_output_vecs_[i], GKMeans::stream(0));
        }
      }


    }

    //put the result to the Y_old
    gk_bdiv<Dtype>(this->N_, this->K_, this->mats_[4]->gpu_data(),
                   this->mats_[5]->gpu_data(), this->mats_[1]->mutable_gpu_data(),
                   GKMeans::stream(0));
  }

  template<typename Dtype>
  void KMeansController<Dtype>::PostProcess(){
    /** Use the cluster center calculated to get cluster labels**/

    int* label_data = this->int_outputs_[0]->mutable_cpu_data();

    // first restart the data provider.
    this->data_providers_[0]->ForceRestart();

    // assign cluster labels
    bool assignment_finished = false;
    while (!assignment_finished){
      size_t batch_num = 0;
      size_t index = this->data_providers_[0]->current_index();

      // getting current index must precede GetData()
      // otherwise the current index in the data provide would be advanced
      this->mats_[0] = this->data_providers_[0]->GetData(batch_num);
      if ( this->data_providers_[0]->current_index()  == 0){
        assignment_finished = true;
      }
      //execute only the maximization functions for all samples
      this->function_input_vecs_[0][0] = this->mats_[0].get();
      this->funcs_[0]->Execute(this->function_input_vecs_[0], this->function_output_vecs_[0], GKMeans::stream(0));
      CUDA_CHECK(cudaStreamSynchronize(GKMeans::stream(0)));

      // copy out cluster labels
      int* out_data = (int*)this->mats_[2]->cpu_data();
      for (size_t i = 0; i < batch_num; i++){
        label_data[i + index] = out_data[i];
      }
    }

    // copy out center data
    memcpy(this->numeric_outputs_[0]->mutable_cpu_data(), this->mats_[1]->cpu_data(), this->mats_[1]->count() * sizeof(Dtype));
  }

  INSTANTIATE_CLASS(KMeansController);

}
