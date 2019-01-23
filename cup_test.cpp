/*
 *  File: cup_test.cpp
 *
 *	This file tests the implementation of the MLP for the regression task
 *  on the ML CUP data set. The model is tested with a k-fold CV on the
 *  entire data set.
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
 *    - <k_fold>
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"
#include "Validation.hpp"

#define MLCUP_TRAIN "Data/ML-CUP18-TR_F.csv"

int main(int argc, char **argv) {
  // Check the number of arguments.
  if (argc < 9) {
    std::cerr << "Usage:" << std::endl
    << argv[0]
    << " <hidden_layer_size> <eta> <alpha> <lambda> <decay>"
    << " <batch_size> <max_iter> <k_fold>" << std::endl;
    return 1;
  }
  // Parse their values.
	int hidden_layer_size = atoi(argv[1]);
	double eta = atof(argv[2]),
	alpha = atof(argv[3]),
	lambda = atof(argv[4]),
  decay = atof(argv[5]);
	int batch_size = atoi(argv[6]),
	max_iter = atoi(argv[7]),
  k = atoi(argv[8]);
	// Read the data from the CSV files.
	arma::mat TR, X_train, Y_train;
  TR.load(MLCUP_TRAIN, arma::csv_ascii);
  X_train = TR.head_cols(10);
  Y_train = TR.tail_cols(2);
  // Build the network.
	MLP r(std::vector<Layer>({
		Layer(hidden_layer_size, X_train.n_cols, sigmoid, sigmoid_d),
		Layer(Y_train.n_cols, hidden_layer_size, identity, identity_d)
	}), eta, alpha, lambda, decay, batch_size, max_iter);
  r.train(X_train, Y_train);
  arma::mat output = r.predict(X_train);
	// Perform the k-fold CV and output the mean score.
	double score = k_fold_CV(r, X_train, Y_train, k, mean_euclidean_error, false);
  std::cout << "Score = " << score << '\n';
  return 0;
}
