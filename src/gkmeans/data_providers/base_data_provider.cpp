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
  shared_ptr<Mat<Dtype>> DataProviderBase<Dtype>::GetData(size_t &num){

    //first join the async task
    num = num_future_.get();

    //also sync the cuda strea
    CUDA_CHECK(cudaStreamSynchronize(data_stream_));

    // get the mat with data filled
    int ready_idx = data_q_.front();
    shared_ptr<Mat<Dtype>> ready_mat = data_slot_vec_[ready_idx];

    // send the id of the mat to the end of the queue
    data_q_.pop_front(); data_q_.push_back(ready_idx);

    // get the mat to be filled (id at the front of the queue)
    Mat<Dtype>* working_mat = data_slot_vec_[data_q_.front()].get();
    // kickout async task (force async launch)
    num_future_ = std::async(std::launch::async, &DataProviderBase::AsyncFunc, this, working_mat);

    // increment the current data cursor.
    current_index_ += num;
    if (current_index_ == round_size_){
      current_index_ = 0;
    }

    return ready_mat;
  }

  template <typename Dtype>
  shared_ptr<Mat<Dtype>> DataProviderBase<Dtype>::SetUp(){
    Mat<Dtype>* data_mat = this->DataSetUp();

    data_slot_vec_.resize(slot_size_);

    for (size_t i = 0; i < slot_size_; ++i){

      //duplicate the output mat for `slot_size_` times
      Mat<Dtype>* mat_ptr = (i == 0)?
                            data_mat:new Mat<Dtype>(data_mat->shape());
      data_slot_vec_[i].reset(mat_ptr);
      data_q_.push_back(i);
    }

    //kick out first async task to fill in data (force async launch)
    Mat<Dtype>* working_mat = data_slot_vec_[data_q_.front()].get();
    num_future_ = std::async(std::launch::async, &DataProviderBase::AsyncFunc, this, working_mat);

    return data_slot_vec_[0];
  }

  template <typename Dtype>
  void DataProviderBase<Dtype>::ForceRestart() {
    //first join the async task
    num_future_.get();
    //also sync the cuda strea
    CUDA_CHECK(cudaStreamSynchronize(data_stream_));

    //reset data cursor
    current_index_ = 0;
    prefetch_index_ = 0;

    Mat<Dtype>* working_mat = data_slot_vec_[data_q_.front()].get();

    // kickout async task (force async launch)
    num_future_ = std::async(std::launch::async, &DataProviderBase::AsyncFunc, this, working_mat);
  }

  INSTANTIATE_CLASS(DataProviderBase);
  INSTANTIATE_CLASS(DummyDataProvider);
}