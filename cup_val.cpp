/*
 *  File: cup_val.cpp
 *
 *	In this file we look for the best MLP hyperparameters
 *  with a grid search combined with a k-fold CV on the training examples.
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"
#include "Validation.hpp"

#define MLCUP_TRAIN_X "Data/ML-CUP18-TR_X.csv"
#define MLCUP_TRAIN_Y "Data/ML-CUP18-TR_Y.csv"

cv_grid_t parameters = {
  .hidden_layer_size_v=std::vector<int>({28}),
  .eta_init_v=std::vector<double>({0.05, 0.07, 0.08}),
  .alpha_v=std::vector<double>({0.1, 0.2}),
  .lambda_v=std::vector<double>({0.007, 0.0075}),
  .decay_v=std::vector<double>({0.45, 0.5}),
  .batch_size_v=std::vector<int>({40, 50}),
  .max_epochs_v=std::vector<int>({2000})
};

int main(int argc, char **argv) {
  // Check the number of arguments.
  if (argc < 3) {
    std::cerr << "Usage:" << std::endl
    << argv[0] << " <k> <par_degree>" << std::endl;
    return 1;
  }
  // Parse their values.
  int k = atoi(argv[1]);
  int par_degree = atoi(argv[2]);
  // Read the data set from the CSV file.
  arma::mat X, Y;
  X.load(MLCUP_TRAIN_X, arma::csv_ascii);
  Y.load(MLCUP_TRAIN_Y, arma::csv_ascii);
  // Perform a grid search with a 10-fold CV.
  std::cout << "Testing " << n_configs(parameters) << 
  " configurations..." << std::endl;
  // Perform grid search CV in verbose mode.
  cv_search_t search_result = grid_search_CV(parameters, X, Y, k,
  par_degree, mean_euclidean_error, true, true, true);
  cv_config_t best_conf = search_result.best_config;
  std::cout << "Best score: " << search_result.best_score << std::endl
  << "Variance: " << search_result.variance << std::endl
  << "Best configuration: " << to_string(best_conf) << std::endl;
  return 0;
}
