#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <cmath>
#include <random>
#include <stdexcept> // For exceptions
#include <iostream>  // For potential debugging output

// Define a type alias for matrices (vector of vectors)
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class NeuralNetwork {
public:
    // Constructor: specifies the number of neurons in each layer (including input and output)
    // Example: {2, 3, 1} means 2 input neurons, 3 hidden neurons, 1 output neuron
    NeuralNetwork(const std::vector<size_t>& layer_sizes, double learning_rate = 0.01);

    // Predict the output for a given input vector
    Vector predict(const Vector& input);

    // Train the network on a single data point (input and target output)
    void train(const Vector& input, const Vector& target);

    // --- Activation Functions ---
    // Sigmoid activation function
    static double sigmoid(double x);
    // Derivative of the sigmoid function
    static double sigmoid_derivative(double x);

    // --- Loss Function ---
    // Mean Squared Error (for a single output vector)
    static double mean_squared_error(const Vector& predicted, const Vector& target);
    // Derivative of Mean Squared Error (for backpropagation)
    static Vector mean_squared_error_derivative(const Vector& predicted, const Vector& target);


private:
    // --- Network Structure ---
    std::vector<size_t> layer_sizes_; // Stores the number of neurons in each layer
    std::vector<Matrix> weights_;     // weights_[i] connects layer i to layer i+1
    std::vector<Vector> biases_;      // biases_[i] is for layer i+1

    // --- Training Parameters ---
    double learning_rate_;

    // --- Internal State (for backpropagation) ---
    std::vector<Vector> layer_outputs_; // Stores outputs of each layer during forward pass (including input)
    std::vector<Vector> layer_inputs_; // Stores weighted inputs to each layer *before* activation

    // --- Helper Methods ---
    // Initialize weights and biases randomly
    void initialize_weights_biases();

    // Perform the forward pass calculation
    Vector forward_pass(const Vector& input);

    // Perform the backpropagation calculation and update weights/biases
    void backpropagate(const Vector& input, const Vector& target);

    // --- Matrix/Vector Operations (Basic implementations) ---
    // These could be replaced by a dedicated linear algebra library for performance
    static Vector multiply(const Matrix& matrix, const Vector& vector);
    static Vector add(const Vector& vec1, const Vector& vec2);
    static Vector subtract(const Vector& vec1, const Vector& vec2);
    static Vector elementwise_multiply(const Vector& vec1, const Vector& vec2);
    static Matrix transpose(const Matrix& matrix);
    static Matrix outer_product(const Vector& vec1, const Vector& vec2); // vec1 * vec2^T
    static Matrix multiply(const Matrix& mat1, double scalar);
    static Matrix subtract(const Matrix& mat1, const Matrix& mat2);
     static Vector multiply(const Vector& vec, double scalar);


};

#endif // NEURAL_NETWORK_H