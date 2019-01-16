/*
 *  File: Error.hpp
 *
 *  This file contains several definitions of functions used to measure
 *  errors during (and after) the training phase.
 */

#ifndef ERROR_H
#define ERROR_H

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

// Prototype of a scoring function: a function that takes two matrices
// (target and output) and returns a score of how accurate is the prediction.
typedef double (*scorer_ptr)(arma::mat, arma::mat);

// Computes the squared error for the current instance.
double squared_error(arma::rowvec target, arma::rowvec output);

// 
arma::rowvec squared_error_d(arma::rowvec target, arma::rowvec output);

// Computes the cross entropy loss for the current instance.
double cross_entropy(arma::rowvec target, arma::rowvec output);

// 
arma::rowvec cross_entropy_d(arma::rowvec target, arma::rowvec output);

// Computes the mean euclidean error between target and output data.
double mean_euclidean_error(arma::mat target, arma::mat output);

// Computes the output accuracy w.r.t. target data.
double accuracy(arma::mat target, arma::mat output);

#endif
