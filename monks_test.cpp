/*
 *  File: monks_test.cpp
 *
 *	This file tests the implementation of the MLP for a classification task
 *	on one of the Monks data set.
 *	The classifier outputs the accuracy on the given test set.
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

int main(int argc, char **argv) {
	// Check the number of arguments.
	if (argc < 9) {
		std::cerr << "Usage:" << std::endl
		<< argv[0]
		<< " <data_set_id> <hidden_layer_size> <eta> <alpha> <lambda>"
		" <decay> <batch_size> <max_iter>" << std::endl;
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
	max_iter = atoi(argv[8]);
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
	// Train the model.
	r.train(X_train, Y_train);
	arma::mat Y_output = r.predict(X_test);
	// Print the accuracy.
	std::cout << "TS accuracy = " << accuracy(Y_test, arma::round(Y_output))
	<< std::endl;
	return 0;
}
