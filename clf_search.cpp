/*
 *  File: clf_search.cpp
 *
 *  In this file we test different configurations of the MLP in order
 *  to maximize the accuracy score on the Monks data sets.
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"
#include "Validation.hpp"

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
	// Read the arguments.
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] <<
		" <data_set_id> <n_configs> <par_degree>" << std::endl;
		return 1;
	}
    int data_set_id = atoi(argv[1]),
	n_configs = atoi(argv[2]),
	par_degree = atoi(argv[3]);
	// Read the data from the CSV files.
	arma::mat X, Y, X_test, Y_test;
	switch (data_set_id) {
		case 1:
			X.load(MONKS1_TRAIN_X, arma::csv_ascii);
			Y.load(MONKS1_TRAIN_Y, arma::csv_ascii);
			X_test.load(MONKS1_TEST_X, arma::csv_ascii);
			Y_test.load(MONKS1_TEST_Y, arma::csv_ascii);
		break;
		case 2:
			X.load(MONKS2_TRAIN_X, arma::csv_ascii);
			Y.load(MONKS2_TRAIN_Y, arma::csv_ascii);
			X_test.load(MONKS2_TEST_X, arma::csv_ascii);
			Y_test.load(MONKS2_TEST_Y, arma::csv_ascii);
		break;
		case 3:
			X.load(MONKS3_TRAIN_X, arma::csv_ascii);
			Y.load(MONKS3_TRAIN_Y, arma::csv_ascii);
			X_test.load(MONKS3_TEST_X, arma::csv_ascii);
			Y_test.load(MONKS3_TEST_Y, arma::csv_ascii);
		break;
		default:
			std::cerr << "Invalid data set ID!" << std::endl;
			return 1;
		break;
	}
	// Create a validation set starting from the given training set.
	// 30% of the records of the original set are reserved for validation.
	cv_partition_t split_p = split_in_two(X, 0.3, true);
	arma::mat X_train = X.rows(split_p.train_ids),
	Y_train = Y.rows(split_p.train_ids),
	X_valid = X.rows(split_p.test_ids),
	Y_valid = Y.rows(split_p.test_ids);
	// Define the space of possible configurations.
	cv_bounds_t params = {
		.hidden_layer_size = std::make_pair(5, 5),
  		.eta_init = std::make_pair(0.1, 0.7),
  		.alpha = std::make_pair(0.2, 0.9),
		.lambda = std::make_pair(0.0001, 0.001),
		.decay = std::make_pair(0.0, 0.5),
		.batch_size = std::make_pair(10, 50),
		.max_epochs = std::make_pair(1000, 5000)
	};
	config_generator cfg_gen(params);
	std::vector<cv_config_t> configs(n_configs);
	std::vector<double> scores(n_configs);
	// Generate n_configs random configurations and try them.
	for (int i = 0; i < n_configs; i++) {
		cv_config_t c = cfg_gen.get_random_config();
		MLP m(std::vector<Layer>({
      		Layer(c.hidden_layer_size, X_train.n_cols, sigmoid, sigmoid_d),
      		Layer(Y_train.n_cols, c.hidden_layer_size, sigmoid, sigmoid_d)
    	}), c.eta_init, c.alpha, c.lambda, c.decay, c.batch_size, 
		c.max_epochs);
		m.train(X_train, Y_train);
		arma::mat Y_out = m.predict(X_valid);
		// Compute the accuracy on the validation set.
		scores.at(i) = accuracy(Y_valid, arma::round(Y_out));
		configs.at(i) = c;
	}
  	std::vector<double>::iterator best = 
  	std::max_element(std::begin(scores), std::end(scores));
  	int x = std::distance(std::begin(scores), best);
	cv_config_t best_conf = configs.at(x);	
  	std::cout << "Best config = " << to_string(best_conf) << std::endl;
	std::cout << "Best score = " << scores.at(x) << std::endl;
	// Model assessment.
	MLP m(std::vector<Layer>({
      	Layer(best_conf.hidden_layer_size, X_train.n_cols, sigmoid, sigmoid_d),
      	Layer(Y_train.n_cols, best_conf.hidden_layer_size, sigmoid, sigmoid_d)
    }), best_conf.eta_init, best_conf.alpha, best_conf.lambda, 
	best_conf.decay, best_conf.batch_size, best_conf.max_epochs);
	m.train(X, Y);
	arma::mat test_out = m.predict(X_test);
	// Compute the accuracy on the test set.
	std::cout << "Test set score = " << 
	accuracy(Y_test, arma::round(test_out)) << std::endl;
	return 0;
}