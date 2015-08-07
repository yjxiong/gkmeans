//
// Created by alex on 8/7/15.
//

#include "gkmeans/common.h"
#include "gkmeans/utils/io.h"
#include "gkmeans/mat.h"

#include "H5Cpp.h"

using namespace H5;

namespace gkmeans {

  template <>
  int WriteDataToHDF5<float>(string file_name, string mat_name, Mat<float>* mat){
    H5::Exception::dontPrint();

    H5File *file;
    try{
      file = new H5File(file_name, H5F_ACC_RDWR);
    }catch(const FileIException&){
      file = new H5File(file_name, H5F_ACC_TRUNC);
    }


    // create h5 boilerplate
    float fill_value = 0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_FLOAT, &fill_value);

    // setup shape info
    vector<size_t> mat_size = mat->shape();
    vector<hsize_t > h5_data_size;
    for (size_t i = 0; i < mat_size.size(); ++i) h5_data_size.push_back(mat_size[i]);
    DataSpace fspace(h5_data_size.size(), h5_data_size.data());
    DataSpace mspace(h5_data_size.size(), h5_data_size.data());

    vector<hsize_t> offset;
    offset.resize(h5_data_size.size()); // an offset vector, it will be zero offset cause we writing all data at once.

    // create dataset and write to file
    DataSet* dataset;
    try {
      dataset = new DataSet(file->createDataSet(
          mat_name, PredType::NATIVE_FLOAT, fspace, plist
      ));
    }catch (const H5::FileIException){
      LOG(WARNING)<<"Overwriting existing HDF5 dataset: "<<mat_name;
      dataset = new DataSet(file->openDataSet(mat_name));
    }

    mspace.selectHyperslab(H5S_SELECT_SET, h5_data_size.data(), offset.data());

    dataset->write(mat->cpu_data(), PredType::NATIVE_FLOAT, mspace, fspace);

    fspace.selectNone();

    delete dataset;
    delete file;

    return 0;
  }

  template <>
  int WriteDataToHDF5<int>(string file_name, string mat_name, Mat<int>* mat){
    H5::Exception::dontPrint();

    H5File *file;
    try{
      file = new H5File(file_name, H5F_ACC_RDWR);
    }catch(const FileIException&){
      file = new H5File(file_name, H5F_ACC_TRUNC);
    }

    // create h5 boilerplate
    int fill_value = 0;
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_INT, &fill_value);

    // setup shape info
    vector<size_t> mat_size = mat->shape();
    vector<hsize_t > h5_data_size;
    for (size_t i = 0; i < mat_size.size(); ++i) h5_data_size.push_back(mat_size[i]);
    DataSpace fspace(h5_data_size.size(), h5_data_size.data());
    DataSpace mspace(h5_data_size.size(), h5_data_size.data());

    vector<hsize_t> offset;
    offset.resize(h5_data_size.size()); // an offset vector, it will be zero offset cause we writing all data at once.

    // create dataset and write to file
    DataSet* dataset;

    try {
      dataset = new DataSet(file->createDataSet(
          mat_name, PredType::NATIVE_INT, fspace, plist
      ));
    }catch (const H5::FileIException){
      LOG(WARNING)<<"Overwriting existing HDF5 dataset: "<<mat_name;
      dataset = new DataSet(file->openDataSet(mat_name));
    }

    mspace.selectHyperslab(H5S_SELECT_SET, h5_data_size.data(), offset.data());

    dataset->write(mat->cpu_data(), PredType::NATIVE_INT, mspace, fspace);

    fspace.selectNone();

    delete dataset;
    delete file;

    return 0;
  }


}