/*
 *  File: cup_test.cpp
 *
 *	This file tests the implementation of the MLP for the regression task
 *  on the ML CUP data set. 
 *
 *	Usage:
 *
 *  cup_test followed by these parameters:
 *
 *    - <hidden_layer_size>
 *    - <eta>
 *    - <alpha>
 *    - <lambda>
 *    - <decay>
 *    - <tol>
 *    - <batch_size>
 *    - <max_iter>
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"

#define MLCUP_TRAIN_X "Data/ML-CUP18-TR_X.csv"
#define MLCUP_TRAIN_Y "Data/ML-CUP18-TR_Y.csv"
#define MLCUP_TEST_X "Data/ML-CUP18-ITS_X.csv"
#define MLCUP_TEST_Y "Data/ML-CUP18-ITS_Y.csv"

int main(int argc, char **argv) {
  // Check the number of arguments.
  if (argc < 8) {
    std::cerr << "Usage:" << std::endl
    << argv[0]
    << " <hidden_layer_size> <eta> <alpha> <lambda> <decay>"
    << " <batch_size> <max_iter>" << std::endl;
    return 1;
  }
  // Parse their values.
	int hidden_layer_size = atoi(argv[1]);
	double eta = atof(argv[2]),
	alpha = atof(argv[3]),
	lambda = atof(argv[4]),
  decay = atof(argv[5]);
	int batch_size = atoi(argv[6]),
	max_iter = atoi(argv[7]);
	// Read the data from the CSV files.
	arma::mat X_train, Y_train, X_test, Y_test;
  X_train.load(MLCUP_TRAIN_X, arma::csv_ascii);
  Y_train.load(MLCUP_TRAIN_Y, arma::csv_ascii);
  X_test.load(MLCUP_TEST_X, arma::csv_ascii);
  Y_test.load(MLCUP_TEST_Y, arma::csv_ascii);
  // Build the network.
	MLP r(std::vector<Layer>({
		Layer(hidden_layer_size, X_train.n_cols, sigmoid, sigmoid_d),
		Layer(Y_train.n_cols, hidden_layer_size, identity, identity_d)
	}), eta, alpha, lambda, decay, batch_size, max_iter);
  // Train on the whole TR set.
  r.train(X_train, Y_train);
  // Test on the internal test set.
  arma::mat Y_out = r.predict(X_test);
  std::cout << "MEE on internal test set: " <<
  mean_euclidean_error(Y_test, Y_out) << std::endl;
  return 0;
}
