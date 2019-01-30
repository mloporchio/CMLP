/*
 *  File: cup_curve.cpp
 *  
 *  This program can be used to generate learning curves for the MLP
 *  using the ML Cup data sets.
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"

#define MLCUP_TRAIN_X "Data/ML-CUP18-TR_X.csv"
#define MLCUP_TRAIN_Y "Data/ML-CUP18-TR_Y.csv"
#define MLCUP_TEST_X "Data/ML-CUP18-ITS_X.csv"
#define MLCUP_TEST_Y "Data/ML-CUP18-ITS_Y.csv"
#define TR_CURVE_FILENAME "train_curve.csv"
#define TS_CURVE_FILENAME "test_curve.csv"

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
    // Train and generate the curves.
    mlp_curve_t c = r.learning_curve(X_train, Y_train, X_test, Y_test, 
    mean_squared_error);
	// Save the curves to CSV files.
    c.train_curve.save(TR_CURVE_FILENAME, arma::csv_ascii);
    c.test_curve.save(TS_CURVE_FILENAME, arma::csv_ascii);
	// Print the scores.
	double train_score = c.train_curve(c.train_curve.size() - 1),
	test_score = c.test_curve(c.test_curve.size() - 1);
	std::cout << "Train score = " << train_score << std::endl
	<< "Test score = " << test_score << std::endl;
    return 0;
}
    