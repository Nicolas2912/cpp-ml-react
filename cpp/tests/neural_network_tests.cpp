#define private public
#include "../neural_network.h"
#undef private

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct TestRunner {
    int total{0};
    int failed{0};

    void expectTrue(bool condition, const std::string& name, const std::string& message = "") {
        ++total;
        if (!condition) {
            ++failed;
            std::cerr << "[FAIL] " << name;
            if (!message.empty()) {
                std::cerr << ": " << message;
            }
            std::cerr << std::endl;
        } else {
            std::cout << "[PASS] " << name << std::endl;
        }
    }

    void expectNear(double actual, double expected, double tolerance, const std::string& name) {
        const double diff = std::fabs(actual - expected);
        expectTrue(diff <= tolerance, name,
                   "expected " + std::to_string(expected) + ", got " + std::to_string(actual) +
                       ", diff " + std::to_string(diff) + ", tolerance " + std::to_string(tolerance));
    }

    template <typename Func>
    void expectThrows(const std::string& name, Func&& func) {
        ++total;
        bool threw_expected = false;
        try {
            func();
        } catch (const std::invalid_argument&) {
            threw_expected = true;
        } catch (const std::exception& e) {
            ++failed;
            std::cerr << "[FAIL] " << name << ": threw unexpected std::exception: " << e.what() << std::endl;
            return;
        } catch (...) {
            ++failed;
            std::cerr << "[FAIL] " << name << ": threw unexpected non-std exception" << std::endl;
            return;
        }

        if (threw_expected) {
            std::cout << "[PASS] " << name << std::endl;
        } else {
            ++failed;
            std::cerr << "[FAIL] " << name << ": expected std::invalid_argument" << std::endl;
        }
    }
};

} // namespace

int main() {
    TestRunner runner;

    runner.expectThrows("constructor rejects too few layers", [] {
        NeuralNetwork nn({1});
        (void)nn;
    });

    runner.expectThrows("predict rejects wrong input size", [] {
        NeuralNetwork nn({2, 1});
        nn.predict({1.0});
    });

    runner.expectNear(NeuralNetwork::sigmoid(0.0), 0.5, 1e-12, "sigmoid at zero");
    runner.expectNear(NeuralNetwork::sigmoid_derivative(0.0), 0.25, 1e-12, "sigmoid derivative at zero");

    {
        Vector predicted{0.0, 0.5};
        Vector target{0.0, 1.0};
        runner.expectNear(NeuralNetwork::mean_squared_error(predicted, target), 0.125, 1e-12, "mean_squared_error computes average");

        Vector derivative = NeuralNetwork::mean_squared_error_derivative(predicted, target);
        runner.expectTrue(derivative.size() == 2 && derivative[0] == 0.0 && std::fabs(derivative[1] + 0.5) < 1e-12,
                          "mean_squared_error_derivative returns predicted-minus-target");
    }

    runner.expectThrows("mean_squared_error rejects mismatched vectors", [] {
        NeuralNetwork::mean_squared_error({1.0}, {1.0, 2.0});
    });

    {
        NeuralNetwork nn({1, 2, 1}, 0.5);
        nn.weights_[0] = {{0.1}, {-0.2}};
        nn.biases_[0] = {0.3, -0.1};
        nn.weights_[1] = {{0.7, -0.3}};
        nn.biases_[1] = {0.05};

        Vector output = nn.forward_pass({0.5});
        runner.expectNear(output.size() == 1 ? output[0] : std::numeric_limits<double>::quiet_NaN(),
                          0.3255825044358744, 1e-9, "forward_pass produces expected output");

        nn.backpropagate({0.5}, {0.1});

        runner.expectNear(nn.weights_[0][0][0], 0.09042694530453856, 1e-9, "backpropagate updates first hidden weight");
        runner.expectNear(nn.biases_[0][0], 0.2808538906090771, 1e-9, "backpropagate updates first hidden bias");
        runner.expectNear(nn.weights_[1][0][0], 0.6338346687008597, 1e-9, "backpropagate updates output weight");
        runner.expectNear(nn.biases_[1][0], -0.06279125221793719, 1e-9, "backpropagate updates output bias");
    }

    {
        NeuralNetwork nn({1, 3, 1}, 0.4);
        nn.weights_[0] = {{0.1}, {0.2}, {-0.1}};
        nn.biases_[0] = {0.0, 0.1, -0.2};
        nn.weights_[1] = {{0.3, -0.4, 0.2}};
        nn.biases_[1] = {0.0};

        std::vector<Vector> inputs{{0.0}, {0.5}, {1.0}};
        std::vector<Vector> targets{{0.0}, {0.25}, {1.0}};

        auto original_prediction = nn.predict({0.5});
        runner.expectTrue(original_prediction.size() == 1, "predict returns single output vector");

        auto predictions = nn.train_for_epochs(inputs, targets, 3, 2);
        runner.expectTrue(predictions.size() == inputs.size(), "train_for_epochs returns prediction per sample");

        Vector post_prediction = nn.predict({0.5});
        runner.expectTrue(post_prediction.size() == 1 && std::fabs(post_prediction[0]) < 10.0,
                          "predict remains numerically stable after training");
    }

    runner.expectThrows("train_for_epochs rejects mismatched dataset sizes", [] {
        NeuralNetwork nn({1, 2, 1});
        std::vector<Vector> inputs{{0.0}};
        std::vector<Vector> targets{{0.0}, {1.0}};
        nn.train_for_epochs(inputs, targets, 1);
    });

    runner.expectThrows("matrix multiply rejects incompatible dimensions", [] {
        Matrix m{{1.0, 2.0}};
        Vector v{1.0};
        NeuralNetwork::multiply(m, v);
    });

    runner.expectThrows("vector add rejects mismatched sizes", [] {
        NeuralNetwork::add({1.0}, {1.0, 2.0});
    });

    runner.expectThrows("matrix subtract rejects mismatched sizes", [] {
        Matrix a{{1.0, 2.0}};
        Matrix b{{1.0}, {2.0}};
        NeuralNetwork::subtract(a, b);
    });

    {
        NeuralNetwork original({1, 8, 1}, 0.1);
        original.train_for_epochs({{0}, {0.5}, {1}}, {{0}, {0.25}, {1}}, 200, 200);
        std::ostringstream saved;
        original.save(saved);
        std::istringstream input(saved.str());
        NeuralNetwork restored = NeuralNetwork::load(input);
        for (double x : {-2.0, 0.137, 0.5, 3.0}) {
            runner.expectNear(restored.predict({x})[0], original.predict({x})[0], 1e-14,
                              "serialized model reproduces actual inference");
        }
    }
    runner.expectThrows("rejects truncated model", [] {
        std::istringstream data("3 1 8 1 sigmoid 0.2");
        NeuralNetwork::load(data);
    });
    runner.expectThrows("rejects model without a known activation", [] {
        std::istringstream data("2 1 1 0.5 0.1");
        NeuralNetwork::load(data);
    });

    runner.expectNear(NeuralNetwork::activate(0.5, Activation::Tanh), std::tanh(0.5), 1e-12, "tanh activation");
    runner.expectNear(NeuralNetwork::activate_derivative(0.5, Activation::Tanh),
                      1.0 - std::tanh(0.5) * std::tanh(0.5), 1e-12, "tanh derivative");
    runner.expectNear(NeuralNetwork::activate(-2.0, Activation::ReLU), 0.0, 1e-12, "relu clamps negatives");
    runner.expectNear(NeuralNetwork::activate(3.0, Activation::ReLU), 3.0, 1e-12, "relu passes positives");
    runner.expectNear(NeuralNetwork::activate_derivative(-2.0, Activation::ReLU), 0.0, 1e-12, "relu derivative negative");
    runner.expectNear(NeuralNetwork::activate_derivative(3.0, Activation::ReLU), 1.0, 1e-12, "relu derivative positive");
    runner.expectTrue(parse_activation("sigmoid") == Activation::Sigmoid &&
                      parse_activation("tanh") == Activation::Tanh &&
                      parse_activation("relu") == Activation::ReLU &&
                      activation_name(Activation::ReLU) == "relu",
                      "activation names round-trip");
    runner.expectThrows("parse_activation rejects unknown names", [] { parse_activation("softmax"); });

    {
        NeuralNetwork nn({4, 6, 1}, 0.01, 7, Activation::Tanh);
        const double limit = std::sqrt(6.0 / (4 + 6));
        bool within = true;
        for (const auto& row : nn.weights_[0]) for (double w : row) within = within && std::fabs(w) <= limit;
        bool zero_bias = true;
        for (double b : nn.biases_[0]) zero_bias = zero_bias && b == 0.0;
        runner.expectTrue(within && zero_bias, "tanh uses Xavier weights and zero biases");
    }
    {
        NeuralNetwork nn({1, 8, 1}, 0.01, 7, Activation::ReLU);
        bool positive_bias = true;
        for (double b : nn.biases_[0]) positive_bias = positive_bias && b > 0.0;
        runner.expectTrue(positive_bias, "relu starts with a small positive bias");
    }

    // Compare backprop's update against a central finite-difference gradient for every activation.
    for (Activation activation : {Activation::Sigmoid, Activation::Tanh, Activation::ReLU}) {
        const double lr = 1e-3;
        NeuralNetwork nn({1, 3, 1}, lr, 42, activation);
        nn.weights_[0] = {{0.4}, {-0.3}, {0.8}};
        nn.biases_[0] = {0.1, 0.2, -0.05};
        nn.weights_[1] = {{0.5, -0.6, 0.7}};
        nn.biases_[1] = {0.05};
        const Vector input{0.7};
        const Vector target{0.2};
        // The engine's loss derivative is (pred - target), the gradient of 0.5 * (pred - target)^2.
        auto loss = [&](NeuralNetwork& net) {
            double diff = net.predict(input)[0] - target[0];
            return 0.5 * diff * diff;
        };
        const double eps = 1e-6;
        NeuralNetwork plus = nn;
        NeuralNetwork minus = nn;
        plus.weights_[0][2][0] += eps;
        minus.weights_[0][2][0] -= eps;
        const double numeric_grad = (loss(plus) - loss(minus)) / (2 * eps);
        const double before = nn.weights_[0][2][0];
        nn.backpropagate(input, target);
        const double analytic_grad = (before - nn.weights_[0][2][0]) / lr;
        runner.expectNear(analytic_grad, numeric_grad, 1e-6,
                          "backprop matches finite differences for " + activation_name(activation));
    }

    for (Activation activation : {Activation::Tanh, Activation::ReLU}) {
        NeuralNetwork original({1, 6, 1}, 0.05, 42, activation);
        std::ostringstream sink;
        auto* cout_buf = std::cout.rdbuf(sink.rdbuf());
        original.train_for_epochs({{0}, {0.5}, {1}}, {{0}, {0.25}, {1}}, 100, 100);
        std::cout.rdbuf(cout_buf);
        std::ostringstream saved;
        original.save(saved);
        std::istringstream input(saved.str());
        NeuralNetwork restored = NeuralNetwork::load(input);
        runner.expectNear(restored.predict({0.3})[0], original.predict({0.3})[0], 1e-14,
                          "serialized " + activation_name(activation) + " model reproduces inference");
    }
    runner.expectThrows("rejects zero layer", [] { NeuralNetwork nn({1, 0, 1}); });
    runner.expectThrows("rejects invalid learning rate", [] { NeuralNetwork nn({1, 2, 1}, -1); });
    runner.expectThrows("rejects invalid epoch count", [] {
        NeuralNetwork nn({1, 2, 1});
        nn.train_for_epochs({{0}}, {{0}}, 0);
    });

    if (runner.failed == 0) {
        std::cout << "\nAll " << runner.total << " neural network tests passed." << std::endl;
        return 0;
    }

    std::cerr << "\n" << runner.failed << " of " << runner.total << " neural network tests failed." << std::endl;
    return 1;
}
