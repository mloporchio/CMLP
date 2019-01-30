/*
 *  File: monks_curve.cpp
 * 
 * 
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"

#define MONKS1_TRAIN_X "Data/monks-1.train.X.csv"
#define MONKS1_TRAIN_Y "Data/monks-1.train.Y.csv"
#define MONKS1_TEST_X "Data/monks-1.test.X.csv"
#define MONKS1_TEST_Y "Data/monks-1.test.Y.csv"
#define MONKS2_TRAIN_X "Data/monks-2.train.X.csv"
#define MONKS2_TRAIN_Y "Data/monks-2.train.Y.csv"
#define MONKS2_TEST_X "Data/monks-2.test.X.csv"
#define MONKS2_TEST_Y "Data/monks-2.test.Y.csv"
#define MONKS3_TRAIN_X "Data/monks-3.train.X.csv"
#define MONKS3_TRAIN_Y "Data/monks-3.train.Y.csv"
#define MONKS3_TEST_X "Data/monks-3.test.X.csv"
#define MONKS3_TEST_Y "Data/monks-3.test.Y.csv"
#define TR_CURVE_FILENAME "Curve/train.csv"
#define TS_CURVE_FILENAME "Curve/test.csv"

int main(int argc, char **argv) {
    // Check the number of arguments.
	if (argc < 10) {
		std::cerr << "Usage:" << std::endl
		<< argv[0]
		<< " <data_set_id> <hidden_layer_size> <eta> <alpha> <lambda>"
		" <decay> <batch_size> <max_iter> <score>" << std::endl
		<< "NOTICE: <score> is either 0 (MSE) or 1 (Accuracy)" << std::endl;
		return 1;
	}
	// Parse their values.
	int data_set_id = atoi(argv[1]),
	hidden_layer_size = atoi(argv[2]);
	double eta = atof(argv[3]),
	alpha = atof(argv[4]),
	lambda = atof(argv[5]),
	decay = atof(argv[6]);
	int batch_size = atoi(argv[7]),
	max_iter = atoi(argv[8]),
	score = atoi(argv[9]);
	// Set the scoring function.
	scorer_ptr score_f;
	switch (score) {
		case 0:
			score_f = mean_squared_error;
		break;
		case 1:
			score_f = accuracy_r;
		break;
		default:
			std::cerr << "Error: invalid scoring function!" << std::endl;
			return 1;
		break;
	}
	// Read the data from the CSV files.
	arma::mat X_train, Y_train, X_test, Y_test;
	switch (data_set_id) {
		case 1:
			X_train.load(MONKS1_TRAIN_X, arma::csv_ascii);
			Y_train.load(MONKS1_TRAIN_Y, arma::csv_ascii);
			X_test.load(MONKS1_TEST_X, arma::csv_ascii);
			Y_test.load(MONKS1_TEST_Y, arma::csv_ascii);
		break;
		case 2:
			X_train.load(MONKS2_TRAIN_X, arma::csv_ascii);
			Y_train.load(MONKS2_TRAIN_Y, arma::csv_ascii);
			X_test.load(MONKS2_TEST_X, arma::csv_ascii);
			Y_test.load(MONKS2_TEST_Y, arma::csv_ascii);
		break;
		case 3:
			X_train.load(MONKS3_TRAIN_X, arma::csv_ascii);
			Y_train.load(MONKS3_TRAIN_Y, arma::csv_ascii);
			X_test.load(MONKS3_TEST_X, arma::csv_ascii);
			Y_test.load(MONKS3_TEST_Y, arma::csv_ascii);
		break;
		default:
			std::cerr << "Invalid data set ID!" << std::endl;
			return 1;
		break;
	}
	// Build the network.
	MLP r(std::vector<Layer>({
		Layer(hidden_layer_size, X_train.n_cols, sigmoid, sigmoid_d),
		Layer(Y_train.n_cols, hidden_layer_size, sigmoid, sigmoid_d)
	}), eta, alpha, lambda, decay, batch_size, max_iter);
	// Train and generate the curves.
    mlp_curve_t c = r.learning_curve(X_train, Y_train, X_test, Y_test, 
    score_f);
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
