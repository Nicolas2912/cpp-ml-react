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
    // With weight decay, backprop must follow the gradient of 0.5 * (pred - target)^2 + 0.5 * l2 * sum(W^2).
    // Check a hidden weight, an output weight, and a bias (biases are not decayed).
    for (Activation activation : {Activation::Sigmoid, Activation::Tanh, Activation::ReLU}) {
        const double lr = 1e-3;
        const double l2 = 0.05;
        NeuralNetwork base({1, 3, 2, 1}, lr, 42, activation);
        base.weights_[0] = {{0.4}, {-0.3}, {0.8}};
        base.biases_[0] = {0.1, 0.2, -0.05};
        base.weights_[1] = {{0.5, -0.6, 0.7}, {-0.2, 0.3, 0.9}};
        base.biases_[1] = {0.05, -0.1};
        base.weights_[2] = {{0.6, -0.4}};
        base.biases_[2] = {0.02};
        base.set_weight_decay(l2);
        const Vector input{0.7};
        const Vector target{0.2};
        auto objective = [&](NeuralNetwork& net) {
            double diff = net.predict(input)[0] - target[0];
            double penalty = 0.0;
            for (const auto& layer : net.weights_) for (const auto& row : layer) for (double w : row) penalty += w * w;
            return 0.5 * diff * diff + 0.5 * l2 * penalty;
        };
        struct Probe { const char* name; std::function<double&(NeuralNetwork&)> param; };
        const std::vector<Probe> probes{
            {"hidden weight", [](NeuralNetwork& n) -> double& { return n.weights_[0][2][0]; }},
            {"middle weight", [](NeuralNetwork& n) -> double& { return n.weights_[1][1][0]; }},
            {"output weight", [](NeuralNetwork& n) -> double& { return n.weights_[2][0][1]; }},
            {"hidden bias", [](NeuralNetwork& n) -> double& { return n.biases_[1][0]; }},
        };
        for (const auto& probe : probes) {
            const double eps = 1e-6;
            NeuralNetwork plus = base, minus = base, step = base;
            probe.param(plus) += eps;
            probe.param(minus) -= eps;
            const double numeric_grad = (objective(plus) - objective(minus)) / (2 * eps);
            const double before = probe.param(step);
            step.backpropagate(input, target);
            const double analytic_grad = (before - probe.param(step)) / lr;
            runner.expectNear(analytic_grad, numeric_grad, 1e-7,
                              std::string("weight-decay backprop matches finite differences for ") + probe.name + " (" +
                                  activation_name(activation) + ")");
        }
    }

    runner.expectThrows("set_weight_decay rejects negative values", [] { NeuralNetwork nn({1, 2, 1}); nn.set_weight_decay(-0.1); });
    runner.expectThrows("set_weight_decay rejects values above 1", [] { NeuralNetwork nn({1, 2, 1}); nn.set_weight_decay(1.5); });
    runner.expectThrows("set_weight_decay rejects NaN", [] { NeuralNetwork nn({1, 2, 1}); nn.set_weight_decay(NAN); });

    // Shared quiet training fixture: a noisy sine where a wide network can memorize noise.
    std::vector<Vector> fit_x, fit_y, val_x, val_y;
    for (int i = 0; i < 24; ++i) {
        const double x = i / 23.0;
        const double noise = (i % 2 ? 0.3 : -0.3) * ((i * 7) % 5) / 4.0;
        fit_x.push_back({x});
        fit_y.push_back({0.5 + 0.4 * std::sin(6 * x) + noise});
    }
    for (int i = 0; i < 8; ++i) {
        const double x = (i + 0.5) / 8.0;
        val_x.push_back({x});
        val_y.push_back({0.5 + 0.4 * std::sin(6 * x)});
    }
    auto quietly = [](const std::function<void()>& body) {
        // Capture progress output at the engine's full precision.
        std::ostringstream sink;
        auto* buffer = std::cout.rdbuf(sink.rdbuf());
        const auto precision = std::cout.precision(17);
        auto restore = [&] {
            std::cout.rdbuf(buffer);
            std::cout.precision(precision);
        };
        try {
            body();
        } catch (...) {
            restore();
            throw;
        }
        restore();
        return sink.str();
    };
    auto squared_weights = [](const NeuralNetwork& net) {
        double total = 0.0;
        for (const auto& layer : net.weights_) for (const auto& row : layer) for (double w : row) total += w * w;
        return total;
    };

    {
        NeuralNetwork plain({1, 16, 1}, 0.1, 42, Activation::Tanh);
        NeuralNetwork decayed({1, 16, 1}, 0.1, 42, Activation::Tanh);
        decayed.set_weight_decay(0.01);
        quietly([&] {
            plain.train_for_epochs(fit_x, fit_y, 300, 300);
            decayed.train_for_epochs(fit_x, fit_y, 300, 300);
        });
        runner.expectTrue(squared_weights(decayed) < 0.8 * squared_weights(plain),
                          "weight decay shrinks the trained weights",
                          std::to_string(squared_weights(decayed)) + " vs " + std::to_string(squared_weights(plain)));
    }

    {
        NeuralNetwork a({1, 8, 1}, 0.1, 42, Activation::Tanh);
        NeuralNetwork b({1, 8, 1}, 0.1, 42, Activation::Tanh);
        Vector legacy;
        TrainingSummary summary;
        const std::string legacy_log = quietly([&] { legacy = a.train_for_epochs(fit_x, fit_y, 50, 10); });
        const std::string fit_log = quietly([&] { summary = b.fit(fit_x, fit_y, 50, 10, {}, {}); });
        runner.expectTrue(legacy == summary.predictions && legacy_log == fit_log && summary.epochs_run == 50 &&
                              summary.best_epoch == 0 && !summary.stopped_early,
                          "fit without validation matches train_for_epochs exactly");
    }

    {
        NeuralNetwork nn({1, 16, 1}, 0.2, 42, Activation::Tanh);
        TrainingSummary summary;
        const std::string log = quietly([&] { summary = nn.fit(fit_x, fit_y, 200, 50, val_x, val_y); });
        double lowest = std::numeric_limits<double>::infinity();
        runner.expectTrue(summary.epochs_run == 200 && !summary.stopped_early && summary.best_epoch >= 1 &&
                              summary.best_epoch <= 200,
                          "fit without patience runs every epoch and tracks the best validation epoch");
        runner.expectTrue(log.find("epoch=50,mse=") == 0 && log.find(",val_mse=") != std::string::npos &&
                              log.find("epoch=200,") != std::string::npos,
                          "progress lines report validation MSE");
        std::istringstream lines(log);
        std::string line;
        while (std::getline(lines, line)) {
            lowest = std::min(lowest, std::stod(line.substr(line.find("val_mse=") + 8)));
        }
        runner.expectTrue(summary.best_validation_mse <= lowest, "best validation MSE is the minimum over all epochs");
        runner.expectNear(nn.dataset_mse(val_x, val_y), std::stod(log.substr(log.rfind("val_mse=") + 8)), 1e-15,
                          "without early stopping the final weights are the last epoch's");
    }

    {
        // Validation targets far from the training targets: fitting the training data keeps making
        // validation worse, so early stopping must fire and restore the best epoch's weights.
        std::vector<Vector> far_y;
        for (const auto& v : val_y) far_y.push_back({-v[0]});
        const int patience = 25;
        NeuralNetwork nn({1, 16, 1}, 0.2, 42, Activation::Tanh);
        TrainingSummary summary;
        const std::string log = quietly([&] { summary = nn.fit(fit_x, fit_y, 2000, 100, val_x, far_y, patience); });
        runner.expectTrue(summary.stopped_early && summary.epochs_run == summary.best_epoch + patience &&
                              summary.epochs_run < 2000,
                          "early stopping fires patience epochs after the best validation epoch",
                          "best " + std::to_string(summary.best_epoch) + ", ran " + std::to_string(summary.epochs_run));
        runner.expectTrue(log.find("epoch=" + std::to_string(summary.epochs_run) + ",") != std::string::npos,
                          "the stopping epoch is reported");
        runner.expectTrue(nn.dataset_mse(val_x, far_y) == summary.best_validation_mse,
                          "early stopping restores the best validation weights exactly");
        Vector restored;
        for (const auto& x : fit_x) restored.push_back(nn.predict(x)[0]);
        runner.expectTrue(restored == summary.predictions, "returned predictions come from the restored weights");
    }

    {
        // A patience longer than the run never stops early but still restores the best epoch.
        NeuralNetwork nn({1, 16, 1}, 0.2, 42, Activation::Tanh);
        TrainingSummary summary;
        quietly([&] { summary = nn.fit(fit_x, fit_y, 120, 120, val_x, val_y, 5000); });
        runner.expectTrue(!summary.stopped_early && summary.epochs_run == 120 &&
                              nn.dataset_mse(val_x, val_y) == summary.best_validation_mse,
                          "a long patience runs every epoch and restores the best weights");
    }

    runner.expectThrows("fit rejects patience without validation data", [&] {
        NeuralNetwork nn({1, 2, 1});
        nn.fit(fit_x, fit_y, 10, 10, {}, {}, 5);
    });
    runner.expectThrows("fit rejects mismatched validation data", [&] {
        NeuralNetwork nn({1, 2, 1});
        nn.fit(fit_x, fit_y, 10, 10, val_x, fit_y);
    });
    runner.expectThrows("fit rejects negative patience", [&] {
        NeuralNetwork nn({1, 2, 1});
        nn.fit(fit_x, fit_y, 10, 10, val_x, val_y, -1);
    });

    // Reference parameters from PyTorch (tests/reference/optimizer_reference.py): three mini-batch
    // steps on a 1-3-2-1 tanh network with weight decay 0.1 on the weights only.
    {
        const std::vector<double> expected_adamw{0.42735996839377088, -0.32719579534279092, 0.82758650672725331, 0.096424821649594511, 0.20269751617503592, -0.050152379701106142, 0.50186469732168093, -0.60619508468446337, 0.72622211629849731, -0.202543858998603, 0.30809359919723178, 0.86887797118240995, 0.046912091350128748, -0.096521518278683133, 0.62614547414026533, -0.375977132751381, 0.016375131293409863};
        const std::vector<double> expected_sgd{0.39936338652396508, -0.29981336563904709, 0.79761661756559821, 0.10058681920333408, 0.19914108734302241, -0.04991488262954534, 0.4989562670681103, -0.59826321148752826, 0.69853813891633054, -0.19974211122850077, 0.29916586542161594, 0.89681090930361473, 0.051006763459523138, -0.10068641157493449, 0.59968774729314578, -0.39802395256948897, 0.021566073371886194};
        const std::vector<std::pair<Vector, Vector>> batches{
            {{0.7, -0.2, 0.4}, {0.2, -0.5, 0.1}},
            {{0.1, 0.9}, {0.6, 0.3}},
            {{-0.8, 0.3, 0.5}, {-0.4, 0.0, 0.7}},
        };
        for (Optimizer optimizer : {Optimizer::Adam, Optimizer::SGD}) {
            NeuralNetwork nn({1, 3, 2, 1}, 0.01, 42, Activation::Tanh);
            nn.weights_[0] = {{0.4}, {-0.3}, {0.8}};
            nn.biases_[0] = {0.1, 0.2, -0.05};
            nn.weights_[1] = {{0.5, -0.6, 0.7}, {-0.2, 0.3, 0.9}};
            nn.biases_[1] = {0.05, -0.1};
            nn.weights_[2] = {{0.6, -0.4}};
            nn.biases_[2] = {0.02};
            nn.set_weight_decay(0.1);
            nn.set_optimizer(optimizer, 3);
            for (const auto& batch : batches) {
                for (size_t i = 0; i < batch.first.size(); ++i) {
                    nn.accumulate_gradients({batch.first[i]}, {batch.second[i]});
                }
                nn.apply_gradients(batch.first.size());
            }
            std::vector<double> actual;
            for (size_t l = 0; l < nn.weights_.size(); ++l) {
                for (const auto& row : nn.weights_[l]) actual.insert(actual.end(), row.begin(), row.end());
                actual.insert(actual.end(), nn.biases_[l].begin(), nn.biases_[l].end());
            }
            const auto& expected = optimizer == Optimizer::Adam ? expected_adamw : expected_sgd;
            double worst = actual.size() == expected.size() ? 0.0 : INFINITY;
            for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
                worst = std::max(worst, std::fabs(actual[i] - expected[i]));
            }
            runner.expectTrue(worst < 1e-12,
                              (optimizer == Optimizer::Adam ? std::string("AdamW") : std::string("mini-batch SGD with L2")) +
                                  " matches PyTorch to 1e-12",
                              "largest difference " + std::to_string(worst));
        }
    }

    runner.expectTrue(parse_optimizer("sgd") == Optimizer::SGD && parse_optimizer("adam") == Optimizer::Adam &&
                          optimizer_name(Optimizer::Adam) == "adam" && optimizer_name(Optimizer::SGD) == "sgd",
                      "optimizer names round-trip");
    runner.expectThrows("parse_optimizer rejects unknown names", [] { parse_optimizer("rmsprop"); });
    runner.expectThrows("set_optimizer rejects a zero batch size", [] {
        NeuralNetwork nn({1, 2, 1});
        nn.set_optimizer(Optimizer::Adam, 0);
    });

    {
        // Full-batch Adam on the fixture: smooth descent and a better fit than the starting point.
        NeuralNetwork nn({1, 16, 1}, 0.01, 42, Activation::Tanh);
        const double before = nn.dataset_mse(fit_x, fit_y);
        nn.set_optimizer(Optimizer::Adam, 1000);
        TrainingSummary summary;
        quietly([&] { summary = nn.fit(fit_x, fit_y, 300, 300, val_x, val_y); });
        runner.expectTrue(nn.dataset_mse(fit_x, fit_y) < 0.5 * before && summary.epochs_run == 300,
                          "full-batch Adam reduces the training loss");
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
