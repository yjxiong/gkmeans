//
// Created by alex on 7/14/15.
//

#ifndef GKMEANS_DATA_PROVIDERS_H
#define GKMEANS_DATA_PROVIDERS_H

#include "gkmeans/common.h"
#include "gkmeans/functions.h"
#include "gkmeans/mat.h"

#include <deque>
#include <future>
using std::deque;
using std::future;

namespace gkmeans{

  /**
   * @brief The base of dataprovider
   * This class implement the "double buffer" scheme
   **/
  template<typename Dtype>
  class DataProviderBase {
  public:

    DataProviderBase(cudaStream_t stream)
        :data_stream_(stream), slot_size_(2){};

    virtual ~DataProviderBase(){
      if (num_future_.valid()){
        num_future_.get();
      }

      for (int i = 0; i < slot_size_; ++i){
        delete data_slot_vec_[i];
      }
    }

    inline virtual const char* DataType(){return "";}

    /**
     * @brief the exposed interface to get ready to process data
     */
    Mat<Dtype>* GetData(int& num);

    /**
     * @brief exposed setup function
     */
    Mat<Dtype>* SetUp();

    /**
     * @brief Async function to prefetch the data
     */
    inline int AsyncFunc(Mat<Dtype> * output_mat){
      int num = this->PrepareData(output_mat);
      output_mat->to_gpu_async(data_stream_);
      return num;
    }

    /**
     * @brief set up internal state and return a mat for the controller to set shapes
     */
    virtual Mat<Dtype>* DataSetUp() {
      LOG(FATAL)<<"DataSetUp() not implemented. Did you type DataSetup?";
      return NULL;
    };

    /**
     * @brief prepare the data and return the number of this batch
     */
    virtual int PrepareData(Mat<Dtype> * output_mat) = 0;


    /**
     * This function can be called explicityly to end the prefetching of the data.
     */
    void EndPrefetching(){
      if (num_future_.valid()){
        num_future_.wait();
      }
    }

  protected:

    future<int> num_future_;
    cudaStream_t data_stream_;

    /**
     * This is where we store the data mats.
     * Usually we have two slots, one serving the functions and one getting filled by PrepareData()
     */
    vector<Mat<Dtype>* > data_slot_vec_;
    int slot_size_;
    deque<int> data_q_;

  };

  /**
   * @brief a dummy data provider doing nothing
   */
  template <typename Dtype>
  class DummyDataProvider: public DataProviderBase<Dtype>{
  public:

    explicit DummyDataProvider(cudaStream_t stream)
      : DataProviderBase<Dtype>(stream){}

    virtual Mat<Dtype>* DataSetUp(){
      size_t n = 1;
      invoke_count_ = 0;
      Mat<Dtype>* mat_ptr = new Mat<Dtype>(vector<size_t>({n}));
      return mat_ptr;
    }
    virtual int PrepareData(Mat<Dtype> * output_mat){
      output_mat->mutable_cpu_data()[0] = invoke_count_;
      invoke_count_++;
      return 1;
    }
  protected:
    int invoke_count_;
  };
}

#endif //GKMEANS_DATA_PROVIDERS_H
