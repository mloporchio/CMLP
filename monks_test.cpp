/*
 *  File: monks_test.cpp
 *
 *	This file tests the implementation of the MLP for a classification task
 *	on one of the Monks data set.
 *	The classifier outputs the accuracy on the given test set.
 */

#include <iostream>
#include <random>
#include "MLP.hpp"
#include "Error.hpp"

#define MAX_TRIALS 100
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
	double total_tr_acc = 0.0, total_ts_acc = 0.0,
	total_tr_mse = 0.0, total_ts_mse = 0.0, 
	mean_tr_acc, mean_ts_acc, mean_tr_mse, mean_ts_mse;
	arma::mat Z(Y_train.n_rows, Y_train.n_cols);
	for (int i = 0; i < MAX_TRIALS; i++) {
		// Build the network.
		MLP r(std::vector<Layer>({
			Layer(hidden_layer_size, X_train.n_cols, sigmoid, sigmoid_d),
			Layer(Y_train.n_cols, hidden_layer_size, sigmoid, sigmoid_d)
		}), eta, alpha, lambda, decay, batch_size, max_iter);
		// Train the model.
		r.train(X_train, Y_train, &Z);
		arma::mat Y_output = r.predict(X_test);
		total_tr_mse += mean_squared_error(Y_train, Z);
		total_ts_mse += mean_squared_error(Y_test, Y_output);
		total_tr_acc += accuracy(Y_train, arma::round(Z));
		total_ts_acc += accuracy(Y_test, arma::round(Y_output));
	}
	mean_tr_mse = total_tr_mse / MAX_TRIALS;
	mean_ts_mse = total_ts_mse / MAX_TRIALS;
	mean_tr_acc = total_tr_acc / MAX_TRIALS;
	mean_ts_acc = total_ts_acc / MAX_TRIALS;
	std::cout << "Mean MSE over TR: " << mean_tr_mse << std::endl;
	std::cout << "Mean MSE over TS: " << mean_ts_mse << std::endl;
	std::cout << "Mean accuracy over TR: " << mean_tr_acc << std::endl;
	std::cout << "Mean accuracy over TS: " << mean_ts_acc << std::endl;
	return 0;
}
