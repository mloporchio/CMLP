/*
 *  File: MLP.hpp
 *
 *  This file contains the implementation of the Multilayer Perceptron class,
 *  the main element of the whole project.
 */

#include "MLP.hpp"
#include "Decay.hpp"
#include <limits>
#include <random>

// Class constructor.
//  - v: vector of layers
//  - eta_init: initial learning rate
//  - alpha: momentum coefficient
//  - lambda: L2 regularization coefficient
//  - decay: learning rate decay coefficient (0 = no decay)
//  - batch_size: number of examples per batch
//  - max_epochs: maximum number of epochs for training
//  - max_unchanged: maximum number of epoch without loss change
//    (optional parameter, default = 10)
//  - tol: floating point tolerance
//    (optional parameter, default = 1E-4)
//  - shuffle: whether to shuffle the examples or not
//    (optional parameter, default = true)
//  - seed: seed for random number initialization 
//    (optional parameter, default = 1)
MLP::MLP(std::vector<Layer> v, double eta_init, double alpha, double lambda,
double decay, int batch_size, int max_epochs, int max_unchanged, 
double tol, bool shuffle, unsigned int seed) 
  : layers(v), gen(seed), dist(DIST_MEAN, DIST_VAR)
{
    this -> l = layers.size();
    this -> eta_init = eta_init;
    this -> eta = eta_init;
    this -> alpha = alpha;
    this -> lambda = lambda;
    this -> decay = decay;
    this -> tol = tol;
    this -> batch_size = batch_size;
    this -> max_epochs = max_epochs;
    this -> max_unchanged = max_unchanged;
    this -> shuffle = shuffle;
}

// Initializes the gradient matrices in each layer with zeros.
void MLP::init_gradients() {
  for (int j = 0; j < l; j++) {
    layers[j].gW.zeros();
    layers[j].gb.zeros();
  }
}

// Initializes weights and biases in each layer with random numbers.
void MLP::init_weights() {
  // We use this lambda function to fill the vectors.
  auto fill_f = [&]() {return dist(gen);};
  for (int j = 0; j < l; j++) {
    layers[j].W.imbue(fill_f);
    layers[j].b.imbue(fill_f);
  }
}

// Performs a training epoch using the minibatch technique.
double MLP::minibatch_train(
  const arma::mat &X, // Input data set.
  const arma::mat &Y, // Target values for the patterns.
  id_vector &ind, // Vector of row indexes, to be used for shuffling.
  arma::mat *output // Pointer to a matrix used to save the training output.
) 
{
  double error = 0.0;
  // Shuffle the elements, if necessary.
  if (shuffle) shuffle_elements(ind);
  // Split the training set into batches.
  std::vector<id_vector> batches = split(ind, batch_size);
  // For each batch of items...
  for (const auto &b : batches) {
    // Initialize the gradients.
    init_gradients();
    // For each item in the current batch...
    for (arma::uword i : b) {
      arma::rowvec x = X.row(i), y = Y.row(i);
      // Forward the pattern across the network.
      layers[0].forward(x);
      for (int j = 1; j < l; j++) layers[j].forward(layers[j-1].y);
      // Store the result in the output matrix, if needed.
      if (output) output -> row(i) = layers[l-1].y;
      // Compute the loss on the current example.
      error += squared_error(y, layers[l-1].y);
      // Backpropagate the error.
      arma::rowvec e = squared_error_d(y, layers[l-1].y);
      layers[l-1].d = e % layers[l-1].dy;
      for (int j = l - 2; j >= 0; j--) {
        layers[j].d = layers[j].dy % (layers[j+1].d * layers[j+1].W);
      }
      // Update the gradients according to the current pattern.
      layers[0].gW += layers[0].d.t() * x;
      layers[0].gb += layers[0].d;
      for (int j = 1; j < l; j++) {
        layers[j].gW += layers[j].d.t() * layers[j-1].y;
        layers[j].gb += layers[j].d;
      }
      // Update the weights of the nodes in each layer.
      for (int j = 0; j < l; j++) {
        layers[j].dW = -eta * ((layers[j].gW / b.size()) 
        + lambda * layers[j].W) + alpha * layers[j].dW;
        layers[j].db = -eta * (layers[j].gb / b.size()) + alpha * layers[j].db;
        layers[j].W += layers[j].dW;
        layers[j].b += layers[j].db;
      }
    }
  }
  return error / X.n_rows;
}

// This method is used to train the network. The training is performed
// using the mini-batch approach, exploiting momentum and L2 regularization.
void MLP::train(const arma::mat &X, const arma::mat &Y) {
  // First of all, check if X and Y have the same number of rows.
  assert(X.n_rows == Y.n_rows);
  int k = 0, count = 0;
  bool converged = false;
  double best_error = std::numeric_limits<double>::infinity();
  // Initialize weights and biases in each layer.
  init_weights();
  // Initialize the learning rate.
  eta = eta_init;
  // Generate the vector containing row indexes.
  id_vector ind = range_v(X.n_rows);
  // Main loop of the training method.
  while (!converged && k < max_epochs) {
    // Train using the minibatch approach.
    double curr_error = minibatch_train(X, Y, ind);
    // Check if convergence has been reached.
    count = (curr_error > best_error - tol) ? (count + 1) : 0;
    if (curr_error < best_error) best_error = curr_error;
    converged = (count >= max_unchanged);
    k++;
    // Update eta for the next epoch.
    eta = invscaling(eta_init, decay, k * X.n_rows);
  }
}

// This function can be used to predict target values for new data.
arma::mat MLP::predict(const arma::mat &X) {
  std::vector<arma::rowvec> output(X.n_rows);
  for (arma::uword i = 0; i < X.n_rows; i++) {
    layers[0].forward(X.row(i));
    for (int j = 1; j < l; j++) layers[j].forward(layers[j-1].y);
    output.at(i) = layers[l-1].y;
  }
  return build_matrix(output);
}

// Generates the learning curves for the given training and validation sets.
mlp_curve_t MLP::learning_curve(
  const arma::mat &X_train, // Training set data.
  const arma::mat &Y_train, // Training set target values.
  const arma::mat &X_val, // Validation set data.
  const arma::mat &Y_val, // Validation set target values.
  scorer_ptr score_f // Scoring function used to evaluate the model.
) 
{
  // This matrix is used to store the output of the training process.
  arma::mat train_output(Y_train.n_rows, Y_train.n_cols);
  // Here we store the scores at each epoch.
  std::vector<double> train_hist, val_hist;
  int k = 0, count = 0;
  bool converged = false;
  double best_error = std::numeric_limits<double>::infinity();
  // Initialize weights and biases in each layer.
  init_weights();
  // Initialize the learning rate.
  eta = eta_init;
  // Generate the vector containing row indexes.
  id_vector ind = range_v(X_train.n_rows);
  // Main loop of the training method.
  while (!converged && k < max_epochs) {
    // Train using the minibatch approach.
    double curr_error = minibatch_train(X_train, Y_train, ind, &train_output);
    // Compute the score on the training set.
    train_hist.push_back(score_f(Y_train, train_output));
    // Compute the score on the validation set.
    arma::mat val_output = predict(X_val);
    val_hist.push_back(score_f(Y_val, val_output));
    // Check if convergence has been reached.
    count = (curr_error > best_error - tol) ? (count + 1) : 0;
    if (curr_error < best_error) best_error = curr_error;
    converged = (count >= max_unchanged);
    k++;
    // Update eta for the next epoch.
    eta = invscaling(eta_init, decay, k * X_train.n_rows);
  }
  mlp_curve_t c = {
    .train_curve = arma::vec(train_hist),
    .val_curve = arma::vec(val_hist)
  };
  return c;
}
