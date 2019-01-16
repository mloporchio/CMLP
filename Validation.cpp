/*
 *  File: Validation.cpp
 *
 *  This file contains the implementation of functions used to perform k-fold
 *  cross-validation over a given data set.
 */

#include <random>
#include <sstream>
#include "Validation.hpp"

// Produces a string describing the configuration.
std::string to_string(cv_config_t c) {
  std::ostringstream s;
  s << "hidden_layer_size=" << c.hidden_layer_size
  << ", eta_init=" << c.eta_init
  << ", alpha=" << c.alpha 
  << ", lambda=" << c.lambda
  << ", decay=" << c.decay
  << ", batch_size=" << c.batch_size << ", max_epochs=" << c.max_epochs;
  return s.str();
}

// Enumerates all the possible configurations of parameters and puts them
// into a single vector.
std::vector<cv_config_t> build_configs(cv_grid_t g) {
  std::vector<cv_config_t> v;
  for (auto h : g.hidden_layer_size_v) {
    for (auto e : g.eta_init_v) {
      for (auto a : g.alpha_v) {
        for (auto l : g.lambda_v) {
          for (auto d : g.decay_v) {
            for (auto b : g.batch_size_v) {
              for (auto m : g.max_epochs_v) {
                v.push_back({
                  .hidden_layer_size=h, 
                  .eta_init=e, 
                  .alpha=a,
                  .lambda=l,
                  .decay=d,
                  .batch_size=b,
                  .max_epochs=m
                });
              }
            }
          }
        }
      }
    }
  }
  return v;
}

// Divides the rows of the data set into 2 groups, the first
// with (1-f) * n elements and the second one with f * n elements.
cv_partition_t split_in_two(const arma::mat &X, double f, bool shuffle) {
  assert(0 < f && f < 1);
  int n = X.n_rows, v_size = f * n, t_size = n - v_size;
  arma::uvec t(t_size), v(v_size);
  id_vector rows = range_v(n);
  // Shuffle the elements before splitting, if required.
  if (shuffle) shuffle_elements(rows);
  for (arma::uword i = 0; i < t_size; i++) t(i) = rows.at(i);
  for (arma::uword i = 0; i < v_size; i++) v(i) = rows.at(t_size + i);
  cv_partition_t p = {.train_ids = t, .test_ids = v};
  return p;
}

// Divides the rows of the data set into k groups.
// Returns a vector of k pairs (train, validation), where each
// of the k groups is considered as a validation set.
std::vector<cv_partition_t> make_partitions(const arma::mat &X, int k, 
bool shuffle) {
  // n = number of patterns in the data set.
  // m = number of elements in each partition.
  int n = X.n_rows, m = n / k;
  id_vector rows = range_v(n);
  if (shuffle) shuffle_elements(rows);
  std::vector<id_vector> splitted = split(rows, m);
  std::vector<cv_partition_t> parts;
  for (int h = splitted.size() - 1; h >= 0; h--) {
    cv_partition_t p;
    id_vector train;
    for (int i = 0; i < splitted.size(); i++) {
      // Skip the h-th vector.
      if (i == h) continue;
      // Fill the train vector.
      for (int j = 0; j < splitted.at(i).size(); j++) 
        train.push_back(splitted.at(i).at(j));  
    }
    // Insert the partition in the result vector.
    parts.push_back({
      .train_ids = arma::uvec(train),
      .test_ids = arma::uvec(splitted.at(h))
    });
  }
  return parts;
}

// Performs a k-fold CV over the given data set.
// Automatically computes the partitioning.
double k_fold_CV(
  MLP m, // Multilayer perceptron.
  const arma::mat &X, // Input data set.
  const arma::mat &Y, // Target features.
  int k, // Number of parts to split the data set into.
  scorer_ptr score_f, // Pointer to a scoring function.
  bool shuffle // Whether to shuffle the patterns or not.
) {
  // Split the data set into k partitions.
  std::vector<cv_partition_t> parts = make_partitions(X, k, shuffle);
  double total_score = 0.0;
  // For each partitioning...
  for (int i = 0; i < parts.size(); i++) {
    cv_partition_t p = parts.at(i);
    // Train the model with the given instances and parameters.
    m.train(X.rows(p.train_ids), Y.rows(p.train_ids));
    // Test the model on the remaining part.
    arma::mat Z = m.predict(X.rows(p.test_ids));
    // Compute the score.
    total_score += score_f(Y.rows(p.test_ids), Z);
  }
  // Return the average score on all the parts.
  return (total_score / parts.size());
}

// Performs a k-fold CV over the given data set.
// The partitioning must have been computed in advance.
double k_fold_CV_prep(
    MLP m,
    const arma::mat &X, // Input data set.
    const arma::mat &Y, // Target features.
    const std::vector<cv_partition_t> &parts, // Partitioning of X.
    scorer_ptr score_f // Pointer to a scoring function.
) {
    double total_score = 0.0;
    for (int i = 0; i < parts.size(); i++) {
        cv_partition_t p = parts.at(i);
        // Train the model with the given instances and parameters.
        m.train(X.rows(p.train_ids), Y.rows(p.train_ids));
        // Test the model on the remaining part.
        arma::mat Z = m.predict(X.rows(p.test_ids));
        // Compute the score.
        total_score += score_f(Y.rows(p.test_ids), Z);
    }
    return (total_score / parts.size());
}

cv_search_t grid_search_CV(
  cv_grid_t parameters, // Parameter grid.
  const arma::mat &X, // Input data set.
  const arma::mat &Y, // Target features.
  int k, // Number of parts to split the data set into.
  int par_degree, // Parallelism degree.
  scorer_ptr score_f, // Pointer to a scoring function.
  bool minimize, // Whether to minimize the score or not.
  bool shuffle // Whether to shuffle the patterns or not.
)
{
  // The partitioning is computed once and for all.
  std::vector<cv_partition_t> parts = make_partitions(X, k, shuffle);
  // Enumerate all the possible configurations in the search space.
  std::vector<cv_config_t> configs = build_configs(parameters);
  // We store one score for each configuration.
  std::vector<double> scores(configs.size());
  // Test each configuration with a given parallelism degree...
  #pragma omp parallel for num_threads(par_degree)
  for (int i = 0; i < configs.size(); i++) {
    cv_config_t c = configs.at(i);
    // Build the model according to the current configuration.
    MLP m(std::vector<Layer>({
      Layer(c.hidden_layer_size, X.n_cols, sigmoid, sigmoid_d),
      Layer(Y.n_cols, c.hidden_layer_size, identity, identity_d)
    }), c.eta_init, c.alpha, c.lambda, c.decay, c.batch_size, c.max_epochs);
    // Do a k-fold CV with the current model.
    scores.at(i) = k_fold_CV_prep(m, X, Y, parts, score_f);
  }
  // This struct is used to store the results.
  cv_search_t search_result;
  // Select the minimum or maximum score among the previously saved ones.
  // Let x be the index of the best score.
  std::vector<double>::iterator best = ((minimize) ?
  std::min_element(std::begin(scores), std::end(scores)) :
  std::max_element(std::begin(scores), std::end(scores)));
  int x = std::distance(std::begin(scores), best);
  search_result.best_score = scores.at(x);
  search_result.best_config = configs.at(x);
  // Return that score.
  return search_result;
}

// Experimental random search with CV.
cv_search_t random_search_CV(cv_bounds_t param_bounds, int max_configs,
const arma::mat &X, const arma::mat &Y, int k, scorer_ptr score_f,
bool minimize, bool shuffle) {
  // Here we store the scores.
  std::vector<double> scores;
  // Here we store the tested configurations.
  std::vector<cv_config_t> configs;
  // Number of tried configurations.
  int n = 0;
  // Build a configuration generator.
  config_generator generator(param_bounds);
  // Compute the partitioning is computed once and for all.
  std::vector<cv_partition_t> parts = make_partitions(X, k, shuffle);
  // Try at most max_configs configurations.
  while (n < max_configs) {
    // Generate a random configuration within the given bounds.
    cv_config_t c = generator.get_random_config();
    // Build the model according to the current configuration.
    MLP m(std::vector<Layer>({
      Layer(c.hidden_layer_size, X.n_cols, sigmoid, sigmoid_d),
      Layer(Y.n_cols, c.hidden_layer_size, identity, identity_d)
    }), c.eta_init, c.alpha, c.lambda, c.decay, c.batch_size, c.max_epochs);
    // Do a k-fold CV with the current model.
    double s = k_fold_CV_prep(m, X, Y, parts, score_f);
    // Save the score to a vector.
    scores.push_back(s);
    configs.push_back(c);
    n++;
  }
  // This struct is used to store the results.
  cv_search_t search_result;
  // Select the minimum or maximum score among the previously saved ones.
  // Let x be the index of the best score.
  std::vector<double>::iterator best = ((minimize) ?
  std::min_element(std::begin(scores), std::end(scores)) :
  std::max_element(std::begin(scores), std::end(scores)));
  int x = std::distance(std::begin(scores), best);
  search_result.best_score = scores.at(x);
  search_result.best_config = configs.at(x);
  // Return that score.
  return search_result;
}
