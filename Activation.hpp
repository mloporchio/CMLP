/*
 *  File: Activation.hpp
 *
 *  This file contains a set of definitions of activation functions
 *  and their derivatives. All the functions are meant to be applied
 *  element-wise on vectors of real numbers.
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

// Shortcut for function pointers.
typedef double (*function_ptr)(double);

inline double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

inline double sigmoid_d(double x) {
  return x * (1.0 - x);
}

inline double tanh_d(double x) {
  double t = tanh(x);
  return 1 - t * t;
}

inline double relu(double x) {
  return ((x > 0) ? x : 0);
}

inline double relu_d(double x) {
  return ((x > 0) ? 1 : 0);
}

inline double identity(double x) {
	return x;
}

inline double identity_d(double x) {
	return 1.0;
}

#endif
