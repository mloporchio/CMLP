/*
 *  File: cv_test.cpp
 *
 *	This file tests the MLP with a k-fold CV on the ML CUP data set.
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"
#include "Validation.hpp"

// 
#define MLCUP_TRAIN_X "Data/ML-CUP18-TR_X.csv"
#define MLCUP_TRAIN_Y "Data/ML-CUP18-TR_Y.csv"

int main(int argc, char **argv) {
  // Check the number of arguments.
  if (argc < 3) {
    std::cerr << "Usage:" << std::endl
    << argv[0] << " <k> <par_degree>" << std::endl;
    return 1;
  }
  // Parse their values.
  int k = atoi(argv[1]),
  par_degree = atoi(argv[2]);
	// Read the data from the CSV files.
	arma::mat X_train, Y_train;
  X_train.load(MLCUP_TRAIN_X, arma::csv_ascii);
  Y_train.load(MLCUP_TRAIN_Y, arma::csv_ascii);
  cv_grid_t parameters = {
    .hidden_layer_size_v=std::vector<int>({5, 10}),
    .eta_init_v=std::vector<double>({0.1, 0.2}),
    .alpha_v=std::vector<double>({0.9}),
    .lambda_v=std::vector<double>({0.0001}),
    .decay_v=std::vector<double>({0.1, 0.2}),
    .batch_size_v=std::vector<int>({30, 40}),
    .max_epochs_v=std::vector<int>({1000})
  };
  cv_search_t search_result = grid_search_CV(parameters, X_train, Y_train, k,
  par_degree, mean_euclidean_error, true, false);
  std::cout << "Best score: " << search_result.best_score << std::endl
  << "Best configuration: " << to_string(search_result.best_config)
  << std::endl;
  
  return 0;
}
