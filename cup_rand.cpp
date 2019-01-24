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

// Bounds for the random search space.
cv_bounds_t bounds = {
    .hidden_layer_size=std::make_pair(10, 10),
    .eta_init=std::make_pair(0.15, 0.15),
    .alpha=std::make_pair(0.18, 0.22),
    .lambda=std::make_pair(0.0015, 0.0025),
    .decay=std::make_pair(0.20, 0.30),
    .batch_size=std::make_pair(30, 40),
    .max_epochs=std::make_pair(2000, 2000),
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