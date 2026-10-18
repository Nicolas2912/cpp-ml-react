#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept> // For exceptions
#include <iostream>  // For potential debugging output

// Define a type alias for matrices (vector of vectors)
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Activation applied to every hidden layer. The output layer is always linear.
enum class Activation { Sigmoid, Tanh, ReLU };

// Parse "sigmoid", "tanh" or "relu"; throws std::invalid_argument otherwise.
Activation parse_activation(const std::string& name);
std::string activation_name(Activation activation);

// SGD updates after every mini-batch with the averaged gradient. Adam keeps running
// averages of the gradient and its square (beta1 0.9, beta2 0.999, epsilon 1e-8).
enum class Optimizer { SGD, Adam };

// Parse "sgd" or "adam"; throws std::invalid_argument otherwise.
Optimizer parse_optimizer(const std::string& name);
std::string optimizer_name(Optimizer optimizer);

// Outcome of a training run. With a validation set, the network tracks the epoch with the
// lowest validation MSE; with early stopping it also restores the weights from that epoch.
struct TrainingSummary {
    Vector predictions;               // Training-set predictions from the final weights
    int epochs_run = 0;               // Epochs actually completed (fewer if stopped early)
    int best_epoch = 0;               // Epoch with the lowest validation MSE (0 without validation)
    double best_validation_mse = NAN; // Validation MSE at best_epoch
    bool stopped_early = false;
};

class NeuralNetwork {
public:
    // Constructor: specifies the number of neurons in each layer (including input and output)
    // Example: {2, 3, 1} means 2 input neurons, 3 hidden neurons, 1 output neuron
    // Sigmoid keeps the original small uniform init; tanh uses Xavier/Glorot and ReLU uses He init.
    NeuralNetwork(const std::vector<size_t>& layer_sizes, double learning_rate = 0.01, unsigned seed = 42,
                  Activation activation = Activation::Sigmoid);

    // Predict the output for a given input vector
    Vector predict(const Vector& input);

    // Train the network on a single data point (input and target output)
    void train(const Vector& input, const Vector& target);

    // Train the network over multiple epochs with periodic loss reporting
    Vector train_for_epochs(
        const std::vector<Vector>& inputs,
        const std::vector<Vector>& targets,
        int epochs,
        int report_every_n_epochs = 10 // Report every 10 epochs by default
    );

    // Train with optional validation data. Validation MSE is measured after every epoch.
    // patience > 0 enables early stopping: training ends once validation MSE has not improved
    // for `patience` epochs, and the weights from the best validation epoch are restored.
    TrainingSummary fit(
        const std::vector<Vector>& inputs,
        const std::vector<Vector>& targets,
        int epochs,
        int report_every_n_epochs,
        const std::vector<Vector>& validation_inputs,
        const std::vector<Vector>& validation_targets,
        int patience = 0
    );

    // Weight decay on W (biases are not decayed); 0, the default, disables it. With SGD this is
    // an L2 penalty: each step follows the gradient of the batch mean of
    // 0.5 * (prediction - target)^2 plus 0.5 * weight_decay * sum(W^2). With Adam it is decoupled
    // weight decay (AdamW): W shrinks by learning_rate * weight_decay * W outside the adaptive update.
    void set_weight_decay(double weight_decay);
    double weight_decay() const { return weight_decay_; }

    // Choose the optimizer and mini-batch size (1 = one update per sample, the default).
    // Batches larger than the dataset use the full dataset. Resets Adam's moment estimates.
    void set_optimizer(Optimizer optimizer, size_t batch_size = 1);

    // Mean squared error of the network over a dataset
    double dataset_mse(const std::vector<Vector>& inputs, const std::vector<Vector>& targets);

    void save(std::ostream& out) const;
    static NeuralNetwork load(std::istream& in);

    // --- Activation Functions ---
    // Sigmoid activation function
    static double sigmoid(double x);
    // Derivative of the sigmoid function
    static double sigmoid_derivative(double x);
    // Apply the given activation (or its derivative) to a pre-activation value z
    static double activate(double z, Activation activation);
    static double activate_derivative(double z, Activation activation);

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
    double weight_decay_ = 0.0;
    std::mt19937 generator_;
    Activation activation_;
    Optimizer optimizer_ = Optimizer::SGD;
    size_t batch_size_ = 1;

    // --- Optimizer State ---
    std::vector<Matrix> grad_weights_; // Gradients summed over the current mini-batch
    std::vector<Vector> grad_biases_;
    std::vector<Matrix> adam_m_weights_, adam_v_weights_; // Adam first and second moments
    std::vector<Vector> adam_m_biases_, adam_v_biases_;
    long adam_step_ = 0;

    // --- Internal State (for backpropagation) ---
    std::vector<Vector> layer_outputs_; // Stores outputs of each layer during forward pass (including input)
    std::vector<Vector> layer_inputs_; // Stores weighted inputs to each layer *before* activation

    // --- Helper Methods ---
    // Initialize weights and biases randomly
    void initialize_weights_biases();

    // Perform the forward pass calculation
    Vector forward_pass(const Vector& input);

    // Perform the backpropagation calculation and update weights/biases from one sample
    void backpropagate(const Vector& input, const Vector& target);

    // Add one sample's loss gradient to the mini-batch accumulators
    void accumulate_gradients(const Vector& input, const Vector& target);

    // Apply the optimizer step for the averaged gradient of `count` samples, then reset it
    void apply_gradients(size_t count);

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