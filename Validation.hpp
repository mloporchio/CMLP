/*
 *  File: Validation.hpp
 *
 *  This file contains the definitions of functions used to perform k-fold
 *  cross-validation over a given data set.
 */

#ifndef VALIDATION_H
#define VALIDATION_H

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include "MLP.hpp"
#include "Error.hpp"
#include "Utils.hpp"

// This is a partitioning of the data set X in two index sets.
// The first index set contains the indexes of the rows of X
// used to train the model, while the second one contains indexes of rows
// used to test the model.
typedef struct {
  arma::uvec train_ids;
  arma::uvec test_ids;
} cv_partition_t;

// This struct represents a configuration of hyperparameters.
typedef struct {
  int hidden_layer_size;
  double eta_init;
  double alpha;
  double lambda;
  double decay;
  int batch_size;
  int max_epochs;
} cv_config_t;

// This is a grid of parameters.
// For each parameter of the MLP we specify a vector of possibilities.
typedef struct {
  std::vector<int> hidden_layer_size_v;
  std::vector<double> eta_init_v;
  std::vector<double> alpha_v;
  std::vector<double> lambda_v;
  std::vector<double> decay_v;
  std::vector<int> batch_size_v;
  std::vector<int> max_epochs_v;
} cv_grid_t;

// This struct is used during the randomized search.
// For each hyperparameter, we specify a pair of values: the minimum
// and the maximum one.
typedef struct {
  std::pair<int, int> hidden_layer_size;
  std::pair<double, double> eta_init;
  std::pair<double, double> alpha;
  std::pair<double, double> lambda;
  std::pair<double, double> decay;
  std::pair<int, int> batch_size;
  std::pair<int, int> max_epochs;
} cv_bounds_t;

// This object is a configuration generator.
// The object has a constructor and a method which is called to produce
// a random configuration with parameters bounded by the specified values.
struct config_generator {
  // The std::random_device is used to produce the seed for 
  // the std::default_random_engine pseudo-random engine.
  std::random_device rd; 
  std::default_random_engine gen;
  std::uniform_int_distribution<> h, b, m;
  std::uniform_real_distribution<> e, a, l, d;
  // Default constructor.
  config_generator(cv_bounds_t bounds) :
    gen(rd()),
    h(bounds.hidden_layer_size.first, bounds.hidden_layer_size.second),
    b(bounds.batch_size.first, bounds.batch_size.second),
    m(bounds.max_epochs.first, bounds.max_epochs.second),
    e(bounds.eta_init.first, bounds.eta_init.second),
    a(bounds.alpha.first, bounds.alpha.second),
    l(bounds.lambda.first, bounds.lambda.second),
    d(bounds.decay.first, bounds.decay.second) {};
  // Method used to obtain a random configuration.
  cv_config_t get_random_config() {
    cv_config_t c = {
      .hidden_layer_size=h(gen),
      .eta_init=e(gen),
      .alpha=a(gen),
      .lambda=l(gen),
      .decay=d(gen),
      .batch_size=b(gen),
      .max_epochs=m(gen)
    };
    return c;
  }
};

// This struct represents the result of a search.
typedef struct {
  // This is the "best" score.
  double best_score;
  // This is the configuration producing that score.
  cv_config_t best_config;
} cv_search_t;


// Produces a string describing the configuration.
std::string to_string(cv_config_t c);

// Returns the number of possible configurations that can be obtained
// from the specified grid.
int n_configs(cv_grid_t g);

// Enumerates all the possible configurations of parameters and puts them
// into a single vector.
std::vector<cv_config_t> build_configs(cv_grid_t g);

// Divides the rows of the data set into 2 groups, the first
// with (1-f) * n elements and the second one with f * n elements.
cv_partition_t split_in_two(const arma::mat &X, double f, bool shuffle);

// Divides the rows of the data set into k groups.
// Returns a vector of k pairs (train, validation), where each
// of the k groups is considered as a validation set.
std::vector<cv_partition_t> make_partitions(const arma::mat &X, int k, 
bool shuffle);

// Performs a k-fold CV over the given data set.
// Automatically computes the partitioning.
double k_fold_CV(MLP m, const arma::mat &X, const arma::mat &Y, int k,
scorer_ptr score_f, bool shuffle);

// Performs a k-fold CV over the given data set.
// The partitioning must have been computed in advance.
double k_fold_CV_prep(MLP m, const arma::mat &X, const arma::mat &Y,
const std::vector<cv_partition_t> &parts, scorer_ptr score_f); 

// Grid search with k-fold cross-validation.
cv_search_t grid_search_CV(cv_grid_t parameters, const arma::mat &X,
const arma::mat &Y, int k, int par_degree, scorer_ptr score_f, 
bool minimize, bool shuffle);

// Random search with k-fold cross-validation.
cv_search_t random_search_CV(cv_bounds_t bounds, int max_configs,
const arma::mat &X, const arma::mat &Y, int k, int par_degree,
scorer_ptr score_f, bool minimize, bool shuffle);

#endif
