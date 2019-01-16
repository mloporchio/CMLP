/*
 *  File: Error.hpp
 *
 *  This file contains the implementation of the functions used to measure
 *  errors during (and after) the training phase.
 */

#include "Error.hpp"

// Computes the squared error for the current instance.
double squared_error(arma::rowvec target, arma::rowvec output) {
  return arma::sum(arma::square(target - output)) / 2;
}

//
arma::rowvec squared_error_d(arma::rowvec target, arma::rowvec output) {
  return output - target;
}

// Computes the cross entropy loss for the current instance.
double cross_entropy(arma::rowvec target, arma::rowvec output) {
  return -arma::sum((target % arma::log(output)) +
  ((1 - target) % arma::log(1 - output)));
}

//
arma::rowvec cross_entropy_d(arma::rowvec target, arma::rowvec output) {
  return ((output - target) / (output % (1 - output)));
}

// Computes the mean euclidean error between target and output data.
double mean_euclidean_error(arma::mat target, arma::mat output) {
  double sum = 0.0;
  for (arma::uword i = 0; i < target.n_rows; i++) {
    sum += arma::norm(target.row(i) - output.row(i), 2);
  }
  return sum / target.n_rows;
}

// Computes the output accuracy w.r.t. target data.
double accuracy(arma::mat target, arma::mat output) {
  // Transform the input data into vectors.
  arma::vec target_v = arma::vectorise(target),
  output_v = arma::vectorise(output);
  // Compute the accuracy.
  arma::uvec x = (target_v == output_v);
  return ((double) arma::accu(x)) / x.n_elem;
}
