/*
 *  File: cup_rand.cpp
 *
 *  In this file we look for the best MLP hyperparameters
 *  with a random search combined with a k-fold CV on the training examples.
 */

#include <iostream>
#include "MLP.hpp"
#include "Error.hpp"
#include "Validation.hpp"

#define MLCUP_TRAIN_X "Data/ML-CUP18-TR_X.csv"
#define MLCUP_TRAIN_Y "Data/ML-CUP18-TR_Y.csv"

cv_bounds_t bounds = {
    .hidden_layer_size=std::make_pair(15, 20),
    .eta_init=std::make_pair(0.1, 0.5),
    .alpha=std::make_pair(0.1, 0.5),
    .lambda=std::make_pair(0.001, 0.002),
    .decay=std::make_pair(0.2, 0.4),
    .batch_size=std::make_pair(20, 40),
    .max_epochs=std::make_pair(3000, 3000)
};

int main(int argc, char **argv) {
    // Check the number of arguments.
    if (argc < 4) {
        std::cerr << "Usage:" << std::endl
        << argv[0] << " <k> <n_conf> <par_degree>" << std::endl;
        return 1;
    }
    // Parse their values.
    int k = atoi(argv[1]);
    int n_conf = atoi(argv[2]);
    int par_degree = atoi(argv[3]);
    // Read the data set from the CSV file.
    arma::mat X, Y;
    X.load(MLCUP_TRAIN_X, arma::csv_ascii);
    Y.load(MLCUP_TRAIN_Y, arma::csv_ascii);
    std::cout << "Testing " << n_conf << " configurations..." << std::endl;
    // Perform random search CV in verbose mode.
    cv_search_t search_result = random_search_CV(bounds, n_conf, X, Y, k,
    par_degree, mean_euclidean_error, true, false, true);
    cv_config_t best_conf = search_result.best_config;
    std::cout << "Best score: " << search_result.best_score << std::endl
    << "Variance: " << search_result.variance << std::endl
    << "Best configuration: " << to_string(best_conf) << std::endl;
    return 0;
}