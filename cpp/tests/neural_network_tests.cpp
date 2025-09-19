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

    if (runner.failed == 0) {
        std::cout << "\nAll " << runner.total << " neural network tests passed." << std::endl;
        return 0;
    }

    std::cerr << "\n" << runner.failed << " of " << runner.total << " neural network tests failed." << std::endl;
    return 1;
}
