/*
 *  File: Utils.hpp
 *
 *  This file contains the implementation of several utility functions used by
 *  the Multilayer Perceptron for both training and prediction.
 */

#include "Utils.hpp"
#include "Layer.hpp"
#include <algorithm>

// This function builds a n * m Armadillo matrix starting from
// a std::vector containing a set of n row vectors of m elements.
arma::mat build_matrix(const std::vector<arma::rowvec> &rows) {
  // Check if vector is not empty.
  // assert(rows.size() > 0);
  arma::mat M = arma::zeros<arma::mat>(rows.size(), rows.at(0).n_elem);
  for (arma::uword i = 0; i < rows.size(); i++) M.row(i) = rows.at(i);
  return M;
}

// Fills a vector with integers in the range [0, k).
id_vector range_v(int k) {
  id_vector v(k);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

// Shuffles a vector of integers according to a random permutation.
void shuffle_elements(id_vector &v) {
  std::random_shuffle(v.begin(), v.end());
}

// Splits a vector into subvectors of k elements each.
std::vector<id_vector> split(const id_vector &v, int k) {
  int i = 0;
  std::vector<id_vector> w;
  while (i + k < v.size()) {
    w.push_back(id_vector(v.begin() + i, v.begin() + i + k));
    i += k;
  }
  w.push_back(id_vector(v.begin() + i, v.end()));
  return w;
}
