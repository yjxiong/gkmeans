//
// Created by alex on 7/15/15.
//

#include "gkmeans/data_providers.h"
#include "H5Cpp.h"

namespace gkmeans {

  template<typename Dtype>
  Mat<Dtype>* HDF5DataProvider<Dtype>::DataSetUp(){

    /** load config*/
    string file_name = GKMeans::get_config("data_file");
    CHECK_NE(file_name, "")<<"Must set a file name to hdf5 data file";
    string data_name = GKMeans::get_config("data_name");
    CHECK_NE(data_name, "")<<"HDF5 dataset name cannot be empty";

    batch_size_ = std::stoul(GKMeans::get_config("batch_size"));
    CHECK_GT(batch_size_, 0);

    /** Open data set and retrieve info */
    h5_file_.reset(new H5::H5File(file_name.c_str(), H5F_ACC_RDONLY));
    h5_dataset_ = h5_file_->openDataSet(data_name.c_str());
    h5_data_space_ = h5_dataset_.getSpace();
    h5_direct_access_space_ = h5_dataset_.getSpace();
    size_t rank = h5_data_space_.getSimpleExtentNdims();

    CHECK_EQ(rank, 2)<<"Currently only support 2 dimensional matrix data";
    dataset_dims_.resize(rank);
    rank = h5_data_space_.getSimpleExtentDims(dataset_dims_.data());

    this->round_size_ = dataset_dims_[0];

    /** Memory buffer for file loading */
    mem_dims_ = dataset_dims_;
    mem_dims_[0] = batch_size_; // memory buffer should have the same size of the batch_size_
    h5_mem_space_ = H5::DataSpace(rank, mem_dims_.data());

    row_dims_ = mem_dims_;
    row_dims_[0] = 1;
    h5_direct_access_mem_space_ = H5::DataSpace(rank, row_dims_.data());

    /** Setup offset */
    offset_.resize(rank);
    offset_.assign(rank, 0);
    direct_access_offset_.resize(rank);
    direct_access_offset_.assign(rank, 0);
    zero_offset_.resize(rank);
    zero_offset_.assign(rank, 0);


    /** setup output mat */
    vector<size_t> mat_dims;
    for(size_t i = 0; i < mem_dims_.size(); ++i){
      mat_dims.push_back(mem_dims_[i]);
    }
    Mat<Dtype>* mat = new Mat<Dtype>(mat_dims);

    direct_access_mat_.reset(new Mat<Dtype>(vector<size_t>(mat_dims.begin() + 1, mat_dims.end())));


    /** try read one part of the dataset*/
    h5_data_space_.selectHyperslab(H5S_SELECT_SET, mem_dims_.data(), offset_.data());
    h5_dataset_.read(mat->mutable_cpu_data(), H5::PredType::NATIVE_FLOAT, h5_mem_space_, h5_data_space_);

    return mat;

  }

  template<typename Dtype>
  size_t HDF5DataProvider<Dtype>::PrepareData(Mat<Dtype> * output_mat){

    /** first determine the number of samples in this batch */
    size_t this_batch = std::min(batch_size_, (this->round_size_ - this->current_index_));

    /** read data */
    offset_[0] = this->prefetch_index_;
    h5_data_space_.selectHyperslab(H5S_SELECT_SET, mem_dims_.data(), offset_.data());
    h5_mem_space_.selectHyperslab(H5S_SELECT_SET, mem_dims_.data(), zero_offset_.data());
    h5_dataset_.read(output_mat->mutable_cpu_data(), H5::PredType::NATIVE_FLOAT, h5_mem_space_, h5_data_space_);

    /** post-processing */
    this->prefetch_index_ += this_batch;
    if (this->prefetch_index_ == this->round_size_){
      this->prefetch_index_ = 0; // rewind if neccesary
    }

    return this_batch;
  }

  template <typename Dtype>
  HDF5DataProvider<Dtype>::~HDF5DataProvider(){
    h5_file_.reset();
  }

  template<typename Dtype>
  Dtype* HDF5DataProvider<Dtype>::DirectAccess(size_t index){
    Dtype* data = direct_access_mat_->mutable_cpu_data();
    direct_access_offset_[0] = index;
    h5_direct_access_space_.selectHyperslab(H5S_SELECT_SET, row_dims_.data(), direct_access_offset_.data());
    h5_direct_access_mem_space_.selectHyperslab(H5S_SELECT_SET, row_dims_.data(), zero_offset_.data());
    h5_mem_space_.selectHyperslab(H5S_SELECT_SET, row_dims_.data(), zero_offset_.data());
    h5_dataset_.read(data, H5::PredType::NATIVE_FLOAT, h5_direct_access_mem_space_, h5_direct_access_space_);
    return data;
  }

  INSTANTIATE_CLASS(HDF5DataProvider);
}