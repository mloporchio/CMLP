/*
 *  File: MLP.hpp
 *
 *  This file contains the definition of the Multilayer Perceptron class,
 *  the main element of the whole project.
 */

#ifndef MLP_H
#define MLP_H

#include "Layer.hpp"
#include "Error.hpp"
#include "Utils.hpp"

// Mean of the distribution used to initialize the weights.
#define DIST_MEAN 0.0
// Variance of the distribution used to initialize the weights.
#define DIST_VAR 1.0

// This struct contains the learning curves for training and validation set.
typedef struct {
  arma::vec train_curve;
  arma::vec test_curve;
} mlp_curve_t;

class MLP {
  private:
    // Here we store the structure of the network.
    std::vector<Layer> layers;
    // Number of layers.
    int l;
    // Initial learning rate.
    double eta_init;
    // Current value of the learning rate.
    double eta;
    // Momentum.
    double alpha;
    // Regularization coefficient.
    double lambda;
    // Learning rate decay. If 0, no decay is used.
    double decay;
    // Error tolerance. Used to check convergence.
    double tol;
    // Number of elements in each batch.
    int batch_size;
    // Maximum number of epochs during training.
    int max_epochs;
    // Maximum number of epochs without change in the loss.
    int max_unchanged;
    // Shuffle the training instances.
    bool shuffle;
    // Random number generator for weights and biases.
    std::default_random_engine gen;
    std::normal_distribution<double> dist;
    // Initializes the gradient matrices in each layer with zeros.
    void init_gradients();
    // Initializes weights and biases in each layer with random numbers.
    void init_weights();
    // Performs a training epoch using the minibatch technique.
    double minibatch_train(const arma::mat &X, const arma::mat &Y, 
    id_vector &ind, arma::mat *output = NULL);
  public:
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
    MLP(std::vector<Layer> v, double eta_init, double alpha, double lambda,
    double decay, int batch_size, int max_epochs, int max_unchanged = 10, 
    double tol = 1E-4, bool shuffle = true, unsigned int seed = 1U);
    // This method is used to train the network. The training is performed
    // using the mini-batch approach, exploiting momentum and L2 regularization.
    void train(const arma::mat &X, const arma::mat &Y);
    // This function can be used to predict target values for new data.
    arma::mat predict(const arma::mat &X);
    // Generates the learning curves for the given training and validation sets.
    mlp_curve_t learning_curve(const arma::mat &X_train, 
    const arma::mat &Y_train, const arma::mat &X_val, const arma::mat &Y_val,
    scorer_ptr score_f); 
};

#endif
