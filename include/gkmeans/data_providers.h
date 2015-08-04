//
// Created by alex on 7/14/15.
//

#ifndef GKMEANS_DATA_PROVIDERS_H
#define GKMEANS_DATA_PROVIDERS_H

#include "gkmeans/common.h"
#include "gkmeans/functions.h"
#include "gkmeans/mat.h"
#include "H5Cpp.h"

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
        :data_stream_(stream), slot_size_(2),
         current_index_(0), prefetch_index_(0){};

    virtual ~DataProviderBase(){
      if (num_future_.valid()){
        num_future_.get();
      }

      data_slot_vec_.clear();
    }

    inline virtual const char* DataType(){return "";}

    /**
     * @brief the exposed interface to get ready to process data
     */
    shared_ptr<Mat<Dtype>> GetData(size_t& num);

    /**
     * @brief exposed setup function
     */
    shared_ptr<Mat<Dtype>> SetUp();

    /**
     * @brief an exposed method for force restarting the data streaming
     */
    void ForceRestart();

    /**
     * @brief Async function to prefetch the data
     */
    inline size_t AsyncFunc(Mat<Dtype> * output_mat){
      size_t num = this->PrepareData(output_mat);
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
     * @brief virtual method to rewind the data cursor of the data provider
     */
    virtual void AdditionalRestartOps(){};

    /**
     * @brief prepare the data and return the number of this batch
     */
    virtual size_t PrepareData(Mat<Dtype> * output_mat) = 0;


    /**
     * This function can be called explicityly to end the prefetching of the data.
     */
    void EndPrefetching(){
      if (num_future_.valid()){
        num_future_.wait();
      }
    }

    /**
     * @brief directly get the i-th sample
     */
    virtual Dtype* DirectAccess(size_t index){return NULL;};

    inline size_t round_size(){return round_size_;}
    inline size_t current_index(){return current_index_;}

  protected:

    future<size_t> num_future_;
    cudaStream_t data_stream_;

    /**
     * This is where we store the data mats.
     * Usually we have two slots, one serving the functions and one getting filled by PrepareData()
     */
    vector<shared_ptr<Mat<Dtype>> > data_slot_vec_;
    size_t slot_size_;
    deque<size_t> data_q_;

    size_t round_size_;
    size_t current_index_;
    size_t prefetch_index_;

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
    virtual size_t PrepareData(Mat<Dtype> * output_mat){
      output_mat->mutable_cpu_data()[0] = invoke_count_;
      invoke_count_++;
      return 1;
    }
  protected:
    int invoke_count_;
  };

  /**
   * @brief a dummy data provider doing nothing
   */
  template <typename Dtype>
  class HDF5DataProvider: public DataProviderBase<Dtype>{
  public:

    explicit HDF5DataProvider(cudaStream_t stream)
        : DataProviderBase<Dtype>(stream){}

    virtual ~HDF5DataProvider();

    virtual Mat<Dtype>* DataSetUp();
    virtual size_t PrepareData(Mat<Dtype> * output_mat);

    virtual Dtype* DirectAccess(size_t index);
  protected:
    shared_ptr<H5::H5File> h5_file_;
    H5::DataSet h5_dataset_;
    H5::DataSpace h5_data_space_;
    H5::DataSpace h5_mem_space_;

    vector<hsize_t> dataset_dims_;
    vector<hsize_t> mem_dims_;
    vector<hsize_t> offset_;

    vector<hsize_t> direct_access_offset_;
    vector<hsize_t> zero_offset_;
    vector<hsize_t> row_dims_;
    H5::DataSpace h5_direct_access_space_;
    H5::DataSpace h5_direct_access_mem_space_;
    shared_ptr<Mat<Dtype>> direct_access_mat_;


    size_t batch_size_;
  };
}

#endif //GKMEANS_DATA_PROVIDERS_H
