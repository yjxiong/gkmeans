//
// Created by alex on 7/9/15.
//

#include <gkmeans/controllers.h>
#include "gkmeans/common.h"
#include "gkmeans/controllers.h"
#include "gkmeans/utils/io.h"

#include "INIReader.h"
#include <chrono>

using gkmeans::GKMeans;

void load_config(char* config_file){
  INIReader reader(config_file);

  CHECK_GE(reader.ParseError(), 0)<<"Cannot parse "<<config_file<<"\n";

  //load configurations

  // input and output, optionally the preset seeds
  string input_data_file_name = reader.Get("Data", "input", "");
  CHECK_NE(input_data_file_name, "")<<"Input data file cannot be empty";
  GKMeans::set_config("data_file", input_data_file_name);

  string input_data_name = reader.Get("Data", "input_data", "data");
  GKMeans::set_config("data_name", input_data_name);

  string preset_center_name = reader.Get("Data", "preset_center_data", "centers");
  if (preset_center_name != ""){
    GKMeans::set_config("preset_center_name", preset_center_name);
  }

  string output_data_file_name = reader.Get("Data", "output", "");
  CHECK_NE(output_data_file_name, "")<<"Output data file cannot be empty";
  GKMeans::set_config("out_data_file", output_data_file_name);

  string output_label_name = reader.Get("Data", "output_label", "label");
  GKMeans::set_config("out_label_name", output_label_name);

  string output_center_name = reader.Get("Data", "output_center", "center");
  GKMeans::set_config("out_center_name", output_center_name);

  // parameters
  int n_cluster = reader.GetInteger("Parameter", "n_cluster", -1);
  CHECK_GT(n_cluster, 0)<<"number of clusters must be greater than 0";
  GKMeans::set_config("n_cluster", std::to_string(n_cluster));

  int batch_size = reader.GetInteger("Parameter", "batch_size", -1);
  CHECK_GT(batch_size, 0)<<"Batch size must be greater than 0";
  GKMeans::set_config("batch_size", std::to_string(batch_size));

  int max_iter = reader.GetInteger("Parameter", "max_iter", -1);
  CHECK_GT(max_iter, 0)<<"Max iteration must be greater than 0";
  GKMeans::set_config("max_iter", std::to_string(max_iter));

  // Seeding
  string seed_type = reader.Get("Seed", "type", "random");
  GKMeans::set_config("seeding_type", seed_type);

  long random_seed = reader.GetInteger("Parameter", "random_seed", -1);
  if (random_seed != -1){
    GKMeans::set_config("random_seed", std::to_string(random_seed));
  }

  if (seed_type == "precomputed"){
    GKMeans::set_config("precomputed_seed_file", reader.Get("Seed", "precomputed", ""));
    GKMeans::set_config("precomputed_seed_name", reader.Get("Seed", "seed_name", ""));
  }


}

int main(int argc, char** argv){

  FLAGS_logtostderr = true;

  google::InitGoogleLogging(argv[0]);

  CHECK_EQ(argc, 2)<<"must specify a config file name";

  LOG(INFO)<<"Loading configurations";

  load_config(argv[1]);

  shared_ptr<gkmeans::Controller<float>> ctrl(new gkmeans::KMeansController<float>());

  ctrl->SetUp();

  LOG(INFO)<<"System setup done. Solving the problem...";

  auto start = std::chrono::high_resolution_clock::now();

  int max_iter = std::stoul(GKMeans::get_config("max_iter"));
  ctrl->Solve(max_iter);

  auto end = std::chrono::high_resolution_clock::now();
  auto diff = end - start;

  LOG(INFO)<<"Solving finished, total time: "<<std::chrono::duration<double, std::milli>(diff).count()<<" ms\n";

  //write results
  string out_file_name = GKMeans::get_config("out_data_file");
  LOG(INFO)<<"Writing results to output file: "<<out_file_name;

  string label_name = GKMeans::get_config("out_label_name");
  DLOG(INFO)<<"output label name: "<<label_name;
  string center_name = GKMeans::get_config("out_center_name");
  DLOG(INFO)<<"output center name: "<<center_name;

  WriteDataToHDF5(out_file_name, label_name, ctrl->int_outputs()[0].get());
  WriteDataToHDF5(out_file_name, center_name, ctrl->numeric_outputs()[0].get());
}