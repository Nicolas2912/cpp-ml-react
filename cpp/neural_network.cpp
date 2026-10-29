#include "neural_network.h"
#include <random>       // For random number generation
#include <stdexcept>    // For exceptions
#include <algorithm>    // For std::transform
#include <cstdlib>      // For std::strtod
#include <iomanip>
#include <limits>
#include <numeric>      // For std::inner_product

Activation parse_activation(const std::string& name) {
    if (name == "sigmoid") return Activation::Sigmoid;
    if (name == "tanh") return Activation::Tanh;
    if (name == "relu") return Activation::ReLU;
    throw std::invalid_argument("Unknown activation '" + name + "' (expected sigmoid, tanh or relu).");
}

std::string activation_name(Activation activation) {
    switch (activation) {
        case Activation::Tanh: return "tanh";
        case Activation::ReLU: return "relu";
        case Activation::Sigmoid:
        default: return "sigmoid";
    }
}

bool parse_finite_double(const std::string& text, double& value) {
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return !text.empty() && end == text.c_str() + text.size() && std::isfinite(value);
}

Optimizer parse_optimizer(const std::string& name) {
    if (name == "sgd") return Optimizer::SGD;
    if (name == "adam") return Optimizer::Adam;
    throw std::invalid_argument("Unknown optimizer '" + name + "' (expected sgd or adam).");
}

std::string optimizer_name(Optimizer optimizer) {
    return optimizer == Optimizer::Adam ? "adam" : "sgd";
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layer_sizes, double learning_rate, unsigned seed,
                             Activation activation)
    : layer_sizes_(layer_sizes), learning_rate_(learning_rate), generator_(seed), activation_(activation) {
    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument("Network must have at least an input and an output layer.");
    }
    if (!std::isfinite(learning_rate) || learning_rate <= 0 || learning_rate > 1) {
        throw std::invalid_argument("Learning rate must be between 0 (exclusive) and 1.");
    }
    if (layer_sizes.size() > 6 || std::any_of(layer_sizes.begin(), layer_sizes.end(), [](size_t n) { return n == 0 || n > 32; })) {
        throw std::invalid_argument("Use at most 6 layers of 1 to 32 neurons.");
    }
    initialize_weights_biases();
}

void NeuralNetwork::initialize_weights_biases() {
    // Sigmoid keeps the original small uniform range so seeded results stay unchanged.
    std::uniform_real_distribution<> dis(-0.5, 0.5); // Distribution for weights
    std::uniform_real_distribution<> bias_dis(0.0, 0.1); // Small positive biases often start well

    weights_.resize(layer_sizes_.size() - 1);
    biases_.resize(layer_sizes_.size() - 1);
    grad_weights_.resize(layer_sizes_.size() - 1);
    grad_biases_.resize(layer_sizes_.size() - 1);

    for (size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        size_t rows = layer_sizes_[i + 1]; // Neurons in current layer
        size_t cols = layer_sizes_[i];     // Neurons in previous layer

        // He (normal, std = sqrt(2 / fan_in)) for ReLU, Xavier/Glorot (uniform,
        // limit = sqrt(6 / (fan_in + fan_out))) for tanh.
        std::normal_distribution<> he_dis(0.0, std::sqrt(2.0 / cols));
        const double xavier_limit = std::sqrt(6.0 / (cols + rows));
        std::uniform_real_distribution<> xavier_dis(-xavier_limit, xavier_limit);

        weights_[i].resize(rows, Vector(cols));
        grad_weights_[i].assign(rows, Vector(cols, 0.0));
        grad_biases_[i].assign(rows, 0.0);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                switch (activation_) {
                    case Activation::ReLU: weights_[i][r][c] = he_dis(generator_); break;
                    case Activation::Tanh: weights_[i][r][c] = xavier_dis(generator_); break;
                    case Activation::Sigmoid:
                    default: weights_[i][r][c] = dis(generator_) * sqrt(1.0 / cols); break;
                }
            }
        }

        // ReLU starts with a small positive bias so fewer units begin "dead"; tanh starts at zero.
        biases_[i].resize(rows);
        for (size_t r = 0; r < rows; ++r) {
            switch (activation_) {
                case Activation::ReLU: biases_[i][r] = 0.01; break;
                case Activation::Tanh: biases_[i][r] = 0.0; break;
                case Activation::Sigmoid:
                default: biases_[i][r] = bias_dis(generator_); break;
            }
        }
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double NeuralNetwork::activate(double z, Activation activation) {
    switch (activation) {
        case Activation::Tanh: return std::tanh(z);
        case Activation::ReLU: return z > 0.0 ? z : 0.0;
        case Activation::Sigmoid:
        default: return sigmoid(z);
    }
}

double NeuralNetwork::activate_derivative(double z, Activation activation) {
    switch (activation) {
        case Activation::Tanh: {
            double t = std::tanh(z);
            return 1.0 - t * t;
        }
        case Activation::ReLU: return z > 0.0 ? 1.0 : 0.0;
        case Activation::Sigmoid:
        default: return sigmoid_derivative(z);
    }
}

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
        if (i < num_hidden_layers) { // Apply the hidden activation (layers 0 to num_hidden_layers-1)
            const Activation activation = activation_;
            std::transform(z.begin(), z.end(), current_output.begin(),
                           [activation](double v) { return activate(v, activation); });
        } else { // Apply linear activation (identity) to output layer (layer num_hidden_layers)
             current_output = z; // a = z
        }
       layer_outputs_[i + 1] = current_output; // Store the output *after* activation/identity
    }

    return current_output; // Final layer's output
}

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
        if (i < num_layers - 2) { // Apply the hidden activation
            const Activation activation = activation_;
            std::transform(z.begin(), z.end(), current_output.begin(),
                           [activation](double v) { return activate(v, activation); });
        } else { // Apply linear activation (identity) to output layer
             current_output = z; // a = z
        }
    }
    return current_output;
}


void NeuralNetwork::accumulate_gradients(const Vector& input, const Vector& target) {
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
        // delta_l = (W_{l+1}^T * delta_{l+1}) * activation'(z_l)
        Matrix W_T = transpose(weights_[i]); // Transpose weights of the layer *ahead*
        Vector propagated_delta = multiply(W_T, deltas[i]); // W^T * delta_next

        Vector layer_input = layer_inputs_[i-1]; // z_l is stored at index i-1
        Vector activation_deriv(layer_input.size());
        const Activation activation = activation_;
        std::transform(layer_input.begin(), layer_input.end(), activation_deriv.begin(),
                       [activation](double v) { return activate_derivative(v, activation); });

        deltas[i - 1] = elementwise_multiply(propagated_delta, activation_deriv);
    }

    // 4. Add this sample's gradients: grad_W_l = delta_l * a_{l-1}^T, grad_b_l = delta_l
    for (size_t i = 0; i < num_layers - 1; ++i) {
        const Vector& current_delta = deltas[i];
        const Vector& prev_layer_output = layer_outputs_[i]; // Activation from the previous layer
        for (size_t r = 0; r < weights_[i].size(); ++r) {
            for (size_t c = 0; c < weights_[i][r].size(); ++c) {
                const double grad = current_delta[r] * prev_layer_output[c];
                grad_weights_[i][r][c] += grad;
            }
            grad_biases_[i][r] += current_delta[r];
        }
    }
}

void NeuralNetwork::backpropagate(const Vector& input, const Vector& target) {
    accumulate_gradients(input, target);
    apply_gradients(1);
}

void NeuralNetwork::apply_gradients(size_t count) {
    // Keeping each product in its own statement stops clang's default FP contraction from fusing
    // them into FMAs, so SGD with batch size 1 and no weight decay matches the original engine bit for bit.
    const double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    double step_size = learning_rate_, correction2 = 1.0;
    if (optimizer_ == Optimizer::Adam) {
        if (adam_m_weights_.empty()) {
            adam_m_weights_ = adam_v_weights_ = grad_weights_;
            adam_m_biases_ = adam_v_biases_ = grad_biases_;
            for (auto* layers : {&adam_m_weights_, &adam_v_weights_})
                for (auto& layer : *layers) for (auto& row : layer) std::fill(row.begin(), row.end(), 0.0);
            for (auto* layers : {&adam_m_biases_, &adam_v_biases_})
                for (auto& layer : *layers) std::fill(layer.begin(), layer.end(), 0.0);
        }
        ++adam_step_;
        step_size = learning_rate_ / (1.0 - std::pow(beta1, static_cast<double>(adam_step_)));
        correction2 = std::sqrt(1.0 - std::pow(beta2, static_cast<double>(adam_step_)));
    }
    // Adam step: m_hat / (sqrt(v_hat) + epsilon), written with the bias corrections folded in.
    auto adam = [&](double grad, double& m, double& v) {
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        const double denominator = std::sqrt(v) / correction2 + epsilon;
        return step_size * (m / denominator);
    };
    const double scale = 1.0 / count;
    for (size_t i = 0; i < weights_.size(); ++i) {
        for (size_t r = 0; r < weights_[i].size(); ++r) {
            for (size_t c = 0; c < weights_[i][r].size(); ++c) {
                double& weight = weights_[i][r][c];
                double grad = count == 1 ? grad_weights_[i][r][c] : grad_weights_[i][r][c] * scale;
                grad_weights_[i][r][c] = 0.0;
                if (optimizer_ == Optimizer::SGD) {
                    if (weight_decay_ > 0) {
                        const double decay = weight_decay_ * weight;
                        grad += decay;
                    }
                    const double step = grad * learning_rate_;
                    weight -= step;
                } else {
                    const double decay = learning_rate_ * weight_decay_ * weight; // AdamW: decoupled decay
                    weight -= adam(grad, adam_m_weights_[i][r][c], adam_v_weights_[i][r][c]) + decay;
                }
            }
            double grad = count == 1 ? grad_biases_[i][r] : grad_biases_[i][r] * scale;
            grad_biases_[i][r] = 0.0;
            if (optimizer_ == Optimizer::SGD) {
                const double bias_step = grad * learning_rate_;
                biases_[i][r] -= bias_step;
            } else {
                biases_[i][r] -= adam(grad, adam_m_biases_[i][r], adam_v_biases_[i][r]);
            }
        }
    }
}

void NeuralNetwork::set_optimizer(Optimizer optimizer, size_t batch_size) {
    if (batch_size == 0 || batch_size > 10000) {
        throw std::invalid_argument("Batch size must be between 1 and 10000.");
    }
    optimizer_ = optimizer;
    batch_size_ = batch_size;
    adam_m_weights_.clear();
    adam_v_weights_.clear();
    adam_m_biases_.clear();
    adam_v_biases_.clear();
    adam_step_ = 0;
}

void NeuralNetwork::set_weight_decay(double weight_decay) {
    if (!std::isfinite(weight_decay) || weight_decay < 0 || weight_decay > 1) {
        throw std::invalid_argument("Weight decay must be between 0 and 1.");
    }
    weight_decay_ = weight_decay;
}

double NeuralNetwork::dataset_mse(const std::vector<Vector>& inputs, const std::vector<Vector>& targets) {
    if (inputs.empty() || inputs.size() != targets.size()) {
        throw std::invalid_argument("Input and target datasets must be non-empty and have the same size.");
    }
    double total = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        total += mean_squared_error(predict(inputs[i]), targets[i]);
    }
    return total / inputs.size();
}


void NeuralNetwork::train(const Vector& input, const Vector& target) {
    // For a single data point, training is just one backpropagation step
    backpropagate(input, target);
    // For batch/stochastic gradient descent, you would accumulate gradients
    // or call this multiple times within an epoch loop.
}


Vector NeuralNetwork::train_for_epochs(
    const std::vector<Vector>& inputs,
    const std::vector<Vector>& targets,
    int epochs,
    int report_every_n_epochs
) {
    return fit(inputs, targets, epochs, report_every_n_epochs, {}, {}).predictions;
}

TrainingSummary NeuralNetwork::fit(
    const std::vector<Vector>& inputs,
    const std::vector<Vector>& targets,
    int epochs,
    int report_every_n_epochs,
    const std::vector<Vector>& validation_inputs,
    const std::vector<Vector>& validation_targets,
    int patience
) {
    if (epochs <= 0 || epochs > 10000 || report_every_n_epochs <= 0) {
        throw std::invalid_argument("Invalid epoch count or reporting interval.");
    }
    if (inputs.empty() || inputs.size() != targets.size()) {
        throw std::invalid_argument("Input and target datasets must be non-empty and have the same size.");
    }
    if (validation_inputs.size() != validation_targets.size()) {
        throw std::invalid_argument("Validation inputs and targets must have the same size.");
    }
    const bool validate = !validation_inputs.empty();
    if (patience < 0 || patience > 10000 || (patience > 0 && !validate)) {
        throw std::invalid_argument("Early stopping needs validation data and a patience of 1 to 10000 epochs.");
    }

    size_t n_samples = inputs.size();
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    TrainingSummary summary;
    std::vector<Matrix> best_weights;
    std::vector<Vector> best_biases;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // Shuffle data for stochasticity (optional but often good)
        std::shuffle(indices.begin(), indices.end(), generator_);

        // One optimizer step per mini-batch, using the batch's average gradient
        for (size_t start = 0; start < n_samples; start += batch_size_) {
            const size_t end = std::min(n_samples, start + batch_size_);
            for (size_t i = start; i < end; ++i) {
                accumulate_gradients(inputs[indices[i]], targets[indices[i]]);
            }
            apply_gradients(end - start);
        }
        summary.epochs_run = epoch;

        double validation_mse = NAN;
        if (validate) {
            validation_mse = dataset_mse(validation_inputs, validation_targets);
            if (!std::isfinite(validation_mse)) {
                throw std::runtime_error("Training diverged. Try a smaller learning rate.");
            }
            if (summary.best_epoch == 0 || validation_mse < summary.best_validation_mse) {
                summary.best_epoch = epoch;
                summary.best_validation_mse = validation_mse;
                if (patience > 0) {
                    best_weights = weights_;
                    best_biases = biases_;
                }
            }
            summary.stopped_early = patience > 0 && epoch - summary.best_epoch >= patience && epoch < epochs;
        }

        // Report loss periodically, on the last epoch, and when stopping early
        if (epoch % report_every_n_epochs == 0 || epoch == epochs || summary.stopped_early) {
            const double current_mse = dataset_mse(inputs, targets);
            if (!std::isfinite(current_mse)) {
                throw std::runtime_error("Training diverged. Try a smaller learning rate.");
            }
            std::cout << "epoch=" << epoch << ",mse=" << current_mse;
            if (validate) std::cout << ",val_mse=" << validation_mse;
            std::cout << std::endl;
        }
        if (summary.stopped_early) break;
    }

    if (patience > 0) {
        weights_ = best_weights;
        biases_ = best_biases;
    }

    // Predictions for the training set from the final (possibly restored) weights.
    // Assuming a single output neuron, matching the frontend.
    summary.predictions.reserve(n_samples);
    for (const auto& input : inputs) {
        summary.predictions.push_back(predict(input)[0]);
    }
    return summary;
}



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

// Round-trip every parameter at double precision. Models stay in server memory.
void NeuralNetwork::save(std::ostream& out) const {
    out << std::setprecision(17) << layer_sizes_.size();
    for (size_t size : layer_sizes_) out << " " << size;
    out << " " << activation_name(activation_);
    for (size_t l = 0; l < weights_.size(); ++l) {
        for (const auto& row : weights_[l]) {
            for (double value : row) out << " " << value;
        }
        for (double value : biases_[l]) out << " " << value;
    }
}

NeuralNetwork NeuralNetwork::load(std::istream& in) {
    size_t count;
    if (!(in >> count) || count < 2 || count > 6) throw std::invalid_argument("Invalid model layers.");
    std::vector<size_t> sizes(count);
    for (auto& size : sizes) {
        if (!(in >> size) || size == 0 || size > 32) throw std::invalid_argument("Invalid model size.");
    }
    std::string activation;
    if (!(in >> activation)) throw std::invalid_argument("Invalid model activation.");
    NeuralNetwork model(sizes, 0.01, 42, parse_activation(activation));
    auto read_value = [&in](double& value) {
        std::string token;
        if (!(in >> token) || !parse_finite_double(token, value)) {
            throw std::invalid_argument("Invalid model parameter.");
        }
    };
    for (size_t l = 0; l < model.weights_.size(); ++l) {
        for (auto& row : model.weights_[l]) for (auto& value : row) read_value(value);
        for (auto& value : model.biases_[l]) read_value(value);
    }
    in >> std::ws;
    if (!in.eof()) throw std::invalid_argument("Unexpected model data.");
    return model;
}
