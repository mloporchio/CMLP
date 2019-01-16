/*
 *  File: Decay.hpp
 *
 *  This file contains a set of functions that can be used to control
 *  the learning rate decay during the training of the MLP.
 */

#ifndef DECAY_H
#define DECAY_H

#include <cmath>

// Exponential learning rate decay.
inline double exp_decay(double eta_init, double d, int k) {
  return eta_init * exp(-d * k);
}

// Invscaling learning rate decay.
inline double invscaling(double eta_init, double d, int t) {
  return eta_init / pow(t, d);
}

#endif
