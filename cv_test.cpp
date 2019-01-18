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
  if (argc < 4) {
    std::cerr << "Usage:" << std::endl
    << argv[0] << " <k> <frac> <par_degree>" << std::endl;
    return 1;
  }
  // Parse their values.
  int k = atoi(argv[1]);
  double frac = atof(argv[2]);
  int par_degree = atoi(argv[3]);
	// Read the data set from the CSV files.
	arma::mat X, Y;
  X.load(MLCUP_TRAIN_X, arma::csv_ascii);
  Y.load(MLCUP_TRAIN_Y, arma::csv_ascii);
  // Split the whole data set into TR, TS set.
  cv_partition_t p = split_in_two(X, frac, true);
  arma::mat X_tr = X.rows(p.train_ids), Y_tr = Y.rows(p.train_ids),
  X_ts = X.rows(p.test_ids), Y_ts = Y.rows(p.test_ids);
  // Perform a grid search with a 10-fold CV.
  cv_grid_t parameters = {
    .hidden_layer_size_v=std::vector<int>({5, 10}),
    .eta_init_v=std::vector<double>({0.1, 0.2}),
    .alpha_v=std::vector<double>({0.9}),
    .lambda_v=std::vector<double>({0.0001}),
    .decay_v=std::vector<double>({0.1, 0.2}),
    .batch_size_v=std::vector<int>({30, 40}),
    .max_epochs_v=std::vector<int>({1000})
  };
  cv_search_t search_result = grid_search_CV(parameters, X_tr, Y_tr, k,
  par_degree, mean_euclidean_error, true, false);
  cv_config_t best_conf = search_result.best_config;
  std::cout << "Best score: " << search_result.best_score << std::endl
  << "Best configuration: " << to_string(best_conf) << std::endl;
  // Re-train on TR and test on the TS.
  MLP m(std::vector<Layer>({
    Layer(best_conf.hidden_layer_size, X_tr.n_cols, sigmoid, sigmoid_d),
    Layer(Y_tr.n_cols, best_conf.hidden_layer_size, identity, identity_d)
  }), best_conf.eta_init, best_conf.alpha, best_conf.lambda, 
	best_conf.decay, best_conf.batch_size, best_conf.max_epochs);
	m.train(X_tr, Y_tr);
	arma::mat test_out = m.predict(X_ts);
  std::cout << "Test score: " << mean_euclidean_error(Y_ts, test_out) 
  << std::endl;
  return 0;
}
