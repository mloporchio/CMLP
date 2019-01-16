/*
 *  File: Utils.hpp
 *
 *  This file contains definitions of several utility functions used by
 *  the Multilayer Perceptron for both training and prediction.
 */

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

// Shortcut for std::vector containing Armadillo's unsigned integers.
typedef std::vector<arma::uword> id_vector;

// This function builds a n * m Armadillo matrix starting from
// a std::vector containing a set of n row vectors of m elements.
arma::mat build_matrix(const std::vector<arma::rowvec> &rows);

// Fills a vector with integers in the range [0, k).
id_vector range_v(int k);

// Shuffles a vector of integers according to a random permutation.
void shuffle_elements(id_vector &v);

// Splits a vector into subvectors of k elements each.
std::vector<id_vector> split(const id_vector &v, int k);
