/*
 *	File: Layer.hpp
 *
 *	This file contains the definition of the generic layer of nodes
 *	for the Multilayer Perceptron architecture.
 */

#ifndef LAYER_H
#define LAYER_H

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include "Activation.hpp"

class Layer {
	private:
		// Number of nodes in the layer.
		int n;
		// Number of connections for each node.
		int k;
		// Pointer to the activation function.
		function_ptr f;
		// Pointer to the activation function derivative.
		function_ptr df;
	public:
		// W is a n * k matrix.
		arma::mat W;
		// dW is a n * k matrix.
		arma::mat dW;
		// b is a vector of n elements.
		arma::rowvec b;
		// db is a vector of n elements.
		arma::rowvec db;
		// gW is a n * k matrix.
		arma::mat gW;
		// gb is a vector of n elements.
		arma::rowvec gb;
		// y is a vector of n elements.
		arma::rowvec y;
		// dy is a vector of n elements.
		arma::rowvec dy;
		// d is a vector of n elements.
		arma::rowvec d;
		// Class constructor.
		Layer(int n, int k, function_ptr af, function_ptr afd);
		// Computes the output of the current layer.
		void forward(const arma::rowvec &x);
		// Returns the number of nodes in this layer.
		int get_n();
		// Returns the number of connections for each node in this layer.
		int get_k();
};

#endif
