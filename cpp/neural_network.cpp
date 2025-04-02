#include "neural_network.h"
#include <random>       // For random number generation
#include <stdexcept>    // For exceptions
#include <algorithm>    // For std::transform
#include <numeric>      // For std::inner_product

// --- Constructor ---
NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layer_sizes, double learning_rate)
    : layer_sizes_(layer_sizes), learning_rate_(learning_rate) {
    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument("Network must have at least an input and an output layer.");
    }
    initialize_weights_biases();
}

// --- Weight and Bias Initialization ---
void NeuralNetwork::initialize_weights_biases() {
    std::random_device rd;
    std::mt19937 gen(rd());
    // He initialization recommended for ReLU, Xavier/Glorot for sigmoid/tanh
    // Using a simple small random range for now
    std::uniform_real_distribution<> dis(-0.5, 0.5); // Distribution for weights
    std::uniform_real_distribution<> bias_dis(0.0, 0.1); // Small positive biases often start well

    weights_.resize(layer_sizes_.size() - 1);
    biases_.resize(layer_sizes_.size() - 1);

    for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        size_t rows = layer_sizes_[i + 1]; // Neurons in current layer
        size_t cols = layer_sizes_[i];     // Neurons in previous layer

        // Initialize weights
        weights_[i].resize(rows, Vector(cols));
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                weights_[i][r][c] = dis(gen) * sqrt(1.0 / cols); // Scaled initialization
            }
        }

        // Initialize biases
        biases_[i].resize(rows);
        for (size_t r = 0; r < rows; ++r) {
            biases_[i][r] = bias_dis(gen);
        }
    }
}

// --- Activation Functions ---
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

// --- Loss Function ---
double NeuralNetwork::mean_squared_error(const Vector& predicted, const Vector& target) {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target vectors must have the same size for MSE.");
    }
    double sum_sq_error = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double error = predicted[i] - target[i];
        sum_sq_error += error * error;
    }
    return sum_sq_error / predicted.size();
}

// Derivative of MSE w.r.t. predicted output (used in backprop delta)
Vector NeuralNetwork::mean_squared_error_derivative(const Vector& predicted, const Vector& target) {
    if (predicted.size() != target.size()) {
        throw std::invalid_argument("Predicted and target vectors must have the same size.");
    }
    Vector derivative = subtract(predicted, target);
    // Often scaled by 2/N, but this constant can be absorbed into the learning rate
    // Keeping it simple: derivative = predicted - target
    return derivative;
}


// --- Forward Pass ---
Vector NeuralNetwork::forward_pass(const Vector& input) {
    if (input.size() != layer_sizes_[0]) {
        throw std::invalid_argument("Input vector size does not match network input layer size.");
    }

    layer_outputs_.assign(layer_sizes_.size(), Vector());
    layer_inputs_.assign(layer_sizes_.size() - 1 , Vector()); // No inputs for the input layer itself

    layer_outputs_[0] = input; // Output of layer 0 is the input itself

    Vector current_output = input;
    size_t num_hidden_layers = layer_sizes_.size() - 2; // Number of layers before the output layer

    // Process layers
    for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) { // Iterate through weights/biases
        // Calculate weighted input: z = W * a_prev + b
        Vector z = add(multiply(weights_[i], current_output), biases_[i]);
        layer_inputs_[i] = z; // Store the input *before* activation

        // Calculate activation (sigmoid for hidden, linear for output)
        current_output.resize(z.size());
        if (i < num_hidden_layers) { // Apply sigmoid to hidden layers (layers 0 to num_hidden_layers-1)
             std::transform(z.begin(), z.end(), current_output.begin(), sigmoid);
        } else { // Apply linear activation (identity) to output layer (layer num_hidden_layers)
             current_output = z; // a = z
        }
       layer_outputs_[i + 1] = current_output; // Store the output *after* activation/identity
    }

    return current_output; // Final layer's output
}

// --- Prediction ---
Vector NeuralNetwork::predict(const Vector& input) {
    // Forward pass without storing intermediate values for training
    if (input.size() != layer_sizes_[0]) {
        throw std::invalid_argument("Input vector size does not match network input layer size.");
    }

    Vector current_output = input;
    size_t num_layers = layer_sizes_.size();
    for (size_t i = 0; i < num_layers - 1; ++i) {
        Vector z = add(multiply(weights_[i], current_output), biases_[i]);
        current_output.resize(z.size());
        if (i < num_layers - 2) { // Apply sigmoid to hidden layers
             std::transform(z.begin(), z.end(), current_output.begin(), sigmoid);
        } else { // Apply linear activation (identity) to output layer
             current_output = z; // a = z
        }
    }
    return current_output;
}


// --- Backpropagation ---
void NeuralNetwork::backpropagate(const Vector& input, const Vector& target) {
    // 1. Perform forward pass to get activations and pre-activation inputs
    Vector predicted_output = forward_pass(input); // This also populates layer_outputs_ and layer_inputs_

    if (target.size() != layer_sizes_.back()) {
        throw std::invalid_argument("Target vector size does not match network output layer size.");
    }

    size_t num_layers = layer_sizes_.size();
    std::vector<Vector> deltas(num_layers - 1); // Error deltas for each layer (excluding input)

    // 2. Calculate delta for the output layer (L)
    // delta_L = (output_L - target) * derivative_of_activation(z_L)
    Vector error_derivative = mean_squared_error_derivative(predicted_output, target);
    Vector last_layer_input = layer_inputs_.back(); // z_L

    // For linear output activation, the derivative is 1
    deltas.back() = error_derivative; // delta_L = (output_L - target) * 1

    // 3. Propagate deltas backwards from L-1 to layer 1
    for (int i = num_layers - 2; i > 0; --i) { // Note: int for loop condition
        // delta_l = (W_{l+1}^T * delta_{l+1}) * sigmoid_derivative(z_l)
        Matrix W_T = transpose(weights_[i]); // Transpose weights of the layer *ahead*
        Vector propagated_delta = multiply(W_T, deltas[i]); // W^T * delta_next

        Vector layer_input = layer_inputs_[i-1]; // z_l is stored at index i-1
        Vector sigmoid_deriv_hidden(layer_input.size());
        std::transform(layer_input.begin(), layer_input.end(), sigmoid_deriv_hidden.begin(), sigmoid_derivative);

        deltas[i - 1] = elementwise_multiply(propagated_delta, sigmoid_deriv_hidden);
    }

    // 4. Update weights and biases using calculated deltas
    for (size_t i = 0; i < num_layers - 1; ++i) {
        const Vector& current_delta = deltas[i];
        const Vector& prev_layer_output = layer_outputs_[i]; // Activation from the previous layer

        // Calculate gradients
        // grad_W_l = delta_l * a_{l-1}^T
        Matrix grad_W = outer_product(current_delta, prev_layer_output);
        // grad_b_l = delta_l
        Vector grad_b = current_delta; // Bias gradient is just the delta

        // Update weights: W = W - learning_rate * grad_W
        weights_[i] = subtract(weights_[i], multiply(grad_W, learning_rate_));

        // Update biases: b = b - learning_rate * grad_b
        biases_[i] = subtract(biases_[i], multiply(grad_b, learning_rate_));
    }
}


// --- Training ---
void NeuralNetwork::train(const Vector& input, const Vector& target) {
    // For a single data point, training is just one backpropagation step
    backpropagate(input, target);
    // For batch/stochastic gradient descent, you would accumulate gradients
    // or call this multiple times within an epoch loop.
}


// --- Train for multiple epochs with reporting ---
Vector NeuralNetwork::train_for_epochs(
    const std::vector<Vector>& inputs,
    const std::vector<Vector>& targets,
    int epochs,
    int report_every_n_epochs
) {
    if (inputs.empty() || inputs.size() != targets.size()) {
        throw std::invalid_argument("Input and target datasets must be non-empty and have the same size.");
    }

    size_t n_samples = inputs.size();
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());

    Vector final_predictions;
    final_predictions.reserve(n_samples);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data for stochasticity (optional but often good)
        std::shuffle(indices.begin(), indices.end(), gen);

        // Train on each sample in the (shuffled) dataset
        for (size_t i = 0; i < n_samples; ++i) {
            size_t idx = indices[i];
            // Simple stochastic gradient descent (one sample at a time)
            backpropagate(inputs[idx], targets[idx]);
            // Note: For larger datasets, mini-batch gradient descent is more common
        }

        // Report loss periodically
        if ((epoch + 1) % report_every_n_epochs == 0 || epoch == epochs - 1) {
            double current_mse = 0.0;
            // Calculate MSE over the *entire* dataset
            for (size_t i = 0; i < n_samples; ++i) {
                 // Use predict, not forward_pass, as we don't need intermediate state here
                Vector prediction = predict(inputs[i]);
                current_mse += mean_squared_error(prediction, targets[i]);
            }
            current_mse /= n_samples;
            std::cout << "epoch=" << (epoch + 1) << ",mse=" << current_mse << std::endl;
        }
    }

    // After training, calculate final predictions for the entire input set
    final_predictions.clear();
    for(const auto& input : inputs) {
        Vector prediction = predict(input);
        // Assuming single output neuron for simplicity based on frontend
        if (!prediction.empty()) {
            final_predictions.push_back(prediction[0]);
        } else {
            final_predictions.push_back(NAN); // Indicate error if prediction failed
        }
    }

    return final_predictions;
}


// --- Matrix/Vector Operations Implementations ---

// Matrix * Vector
Vector NeuralNetwork::multiply(const Matrix& matrix, const Vector& vector) {
    if (matrix.empty() || matrix[0].size() != vector.size()) {
        throw std::invalid_argument("Matrix columns must match vector size for multiplication.");
    }
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    Vector result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
        // Alternative using inner_product:
        // result[i] = std::inner_product(matrix[i].begin(), matrix[i].end(), vector.begin(), 0.0);
    }
    return result;
}

// Vector + Vector
Vector NeuralNetwork::add(const Vector& vec1, const Vector& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for addition.");
    }
    Vector result(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::plus<double>());
    return result;
}

// Vector - Vector
Vector NeuralNetwork::subtract(const Vector& vec1, const Vector& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for subtraction.");
    }
    Vector result(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::minus<double>());
    return result;
}

// Vector .* Vector (Element-wise multiplication)
Vector NeuralNetwork::elementwise_multiply(const Vector& vec1, const Vector& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size for element-wise multiplication.");
    }
    Vector result(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), result.begin(), std::multiplies<double>());
    return result;
}

// Matrix Transpose
Matrix NeuralNetwork::transpose(const Matrix& matrix) {
    if (matrix.empty()) {
        return Matrix();
    }
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    Matrix result(cols, Vector(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

// Outer Product (vector * vector^T) -> Matrix
Matrix NeuralNetwork::outer_product(const Vector& vec1, const Vector& vec2) {
    size_t rows = vec1.size();
    size_t cols = vec2.size();
    Matrix result(rows, Vector(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = vec1[i] * vec2[j];
        }
    }
    return result;
}

// Matrix * Scalar
Matrix NeuralNetwork::multiply(const Matrix& mat, double scalar) {
    Matrix result = mat; // Copy
    for (auto& row : result) {
        std::transform(row.begin(), row.end(), row.begin(),
                       [scalar](double val){ return val * scalar; });
    }
    return result;
}

// Matrix - Matrix
Matrix NeuralNetwork::subtract(const Matrix& mat1, const Matrix& mat2) {
     if (mat1.size() != mat2.size() || (!mat1.empty() && mat1[0].size() != mat2[0].size())) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
    }
    Matrix result = mat1; // Copy
    for (size_t i = 0; i < result.size(); ++i) {
         std::transform(result[i].begin(), result[i].end(), mat2[i].begin(), result[i].begin(), std::minus<double>());
    }
    return result;
}

// Vector * Scalar
Vector NeuralNetwork::multiply(const Vector& vec, double scalar) {
    Vector result = vec; // Copy
    std::transform(result.begin(), result.end(), result.begin(),
                   [scalar](double val){ return val * scalar; });
    return result;
}