//
// Created by alex on 7/14/15.
//



#include "gkmeans/data_providers.h"
#include "../../../../../../../usr/include/c++/4.7/future"

namespace gkmeans{

  /**
   * Data providers does not take any input mats.
   * This function will initial a
   */
  template <typename Dtype>
  Mat<Dtype>* DataProviderBase<Dtype>::GetData(size_t &num){

    //first join the async task
    num = num_future_.get();

    //also sync the cuda strea
    CUDA_CHECK(cudaStreamSynchronize(data_stream_));

    // get the mat with data filled
    int ready_idx = data_q_.front();
    Mat<Dtype>* ready_mat = data_slot_vec_[ready_idx];

    // send the id of the mat to the end of the queue
    data_q_.pop_front(); data_q_.push_back(ready_idx);

    // get the mat to be filled (id at the front of the queue)
    Mat<Dtype>* working_mat = data_slot_vec_[data_q_.front()];

    // kickout async task (force async launch)
    num_future_ = std::async(std::launch::async, &DataProviderBase::AsyncFunc, this, working_mat);

    current_index_ += num;
    if (current_index_ == round_size_){
      current_index_ = 0;
    }

    return ready_mat;
  }

  template <typename Dtype>
  Mat<Dtype>* DataProviderBase<Dtype>::SetUp(){
    Mat<Dtype>* data_mat = this->DataSetUp();

    for (size_t i = 0; i < slot_size_; ++i){

      //duplicate the output mat for `slot_size_` times
      Mat<Dtype>* mat_ptr = (i == 0)?
                            data_mat:new Mat<Dtype>(data_mat->shape());
      data_slot_vec_.push_back(mat_ptr);
      data_q_.push_back(i);
    }

    //kick out first async task to fill in data (force async launch)
    Mat<Dtype>* working_mat = data_slot_vec_[data_q_.front()];
    num_future_ = std::async(std::launch::async, &DataProviderBase::AsyncFunc, this, working_mat);

    return data_mat;
  }

  INSTANTIATE_CLASS(DataProviderBase);
  INSTANTIATE_CLASS(DummyDataProvider);
}