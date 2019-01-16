/*
 *	File: Layer.hpp
 *
 *	This file contains the implementation of the generic layer of nodes
 *	for the Multilayer Perceptron architecture.
 */

#include <random>
#include <iostream>
#include "Layer.hpp"

// Class constructor.
Layer::Layer(int n, int k, function_ptr af, function_ptr afd) :
	// Initialize the matrices and vectors.
	W(n, k, arma::fill::zeros), b(n, arma::fill::zeros),
	dW(n, k, arma::fill::zeros), db(n, arma::fill::zeros),
	gW(n, k, arma::fill::zeros), gb(n, arma::fill::zeros),
	y(n, arma::fill::zeros), dy(n, arma::fill::zeros), d(n, arma::fill::zeros)
{
	this -> n = n;
	this -> k = k;
	this -> f = af;
	this -> df = afd;
}

// Computes the output of the current layer.
void Layer::forward(const arma::rowvec &x) {
	// Each element of the output is computed in the following way:
	//	y[j] = activation_function(sum_{i=0}^{k-1}{w[j][i] * x[i]} + b[j]).
	for (arma::uword j = 0; j < y.n_elem; j++) {
		y(j) = f(arma::dot(W.row(j), x) + b(j));
		// Compute also the derivative (to be used later on).
		dy(j) = df(y(j));
	}
}

// Returns the number of nodes in this layer.
int Layer::get_n() {
	return this -> n;
}

// Returns the number of connections for each node in this layer.
int Layer::get_k() {
	return this -> k;
}
