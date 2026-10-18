#include "linear_regression.h"
#include "neural_network.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

std::vector<double> parseVector(const std::string& text) {
    if (text.empty()) return {};
    if (text.back() == ',') throw std::invalid_argument("Empty numeric value.");
    std::vector<double> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        size_t consumed = 0;
        double value = std::stod(token, &consumed);
        if (consumed != token.size() || !std::isfinite(value)) {
            throw std::invalid_argument("Expected finite comma-separated numbers.");
        }
        values.push_back(value);
        if (values.size() > 1200) throw std::invalid_argument("Too many data points.");
    }
    return values;
}

std::vector<size_t> parseLayerSizes(const std::string& text) {
    if (text.empty() || text.back() == '-') throw std::invalid_argument("Invalid layers.");
    std::vector<size_t> sizes;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, '-')) {
        if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("Layer sizes must be positive integers.");
        }
        unsigned long size = std::stoul(token);
        if (size == 0 || size > 32) throw std::invalid_argument("Use 1 to 32 neurons per layer.");
        sizes.push_back(size);
    }
    if (sizes.size() < 2 || sizes.size() > 6 || sizes.front() != 1 || sizes.back() != 1) {
        throw std::invalid_argument("Use 2 to 6 layers with one input and one output.");
    }
    return sizes;
}

struct TrainingOptions {
    double weight_decay = 0.0;
    int patience = 0;
    Optimizer optimizer = Optimizer::SGD;
    size_t batch_size = 1;
};

// Parse one key=value option: weight_decay=<0..1>, patience=<epochs>, optimizer=sgd|adam, batch=<size>.
void parseTrainingOption(const std::string& text, TrainingOptions& options) {
    const size_t at = text.find('=');
    if (at == std::string::npos) throw std::invalid_argument("Expected a key=value training option.");
    const std::string key = text.substr(0, at), value = text.substr(at + 1);
    if (key == "optimizer") {
        options.optimizer = parse_optimizer(value);
        return;
    }
    const auto numbers = parseVector(value);
    if (numbers.size() != 1) throw std::invalid_argument("Option " + key + " needs one number.");
    const double number = numbers[0];
    const bool whole = std::floor(number) == number;
    if (key == "weight_decay" && number >= 0 && number <= 1) {
        options.weight_decay = number;
    } else if (key == "patience" && whole && number >= 0 && number <= 10000) {
        options.patience = static_cast<int>(number);
    } else if (key == "batch" && whole && number >= 1 && number <= 10000) {
        options.batch_size = static_cast<size_t>(number);
    } else {
        throw std::invalid_argument("Invalid training option " + text + ".");
    }
}

std::vector<double> readAndParseVectorFromStdin() {
    std::string line;
    std::getline(std::cin, line);
    return parseVector(line);
}

void printVector(const Vector& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        if (!std::isfinite(values[i])) throw std::runtime_error("Model produced a non-finite prediction.");
        std::cout << values[i];
    }
}

void printUsage(const char* name) {
    std::cerr << "Usage: " << name << " lr_train | lr_predict <slope> <intercept> <x> | "
              << "nn_train_predict <layers> <learning-rate> <epochs> [sigmoid|tanh|relu] [weight_decay=<0..1>] "
              << "[patience=<epochs>] [optimizer=sgd|adam] [batch=<size>] | nn_predict\n"
              << "Training reads X and Y CSV lines; NN training optionally reads evaluation X,\n"
              << "then validation X and Y lines. A patience above 0 enables early stopping.\n"
              << "nn_predict reads a serialized model line followed by an X CSV line.\n";
}

#ifndef UNIT_TESTING
int main(int argc, char** argv) {
    std::cout << std::setprecision(17);
    try {
        if (argc < 2) throw std::invalid_argument("Choose an operation.");
        const std::string operation = argv[1];
        if (operation == "lr_predict" && argc == 5) {
            const double slope = std::stod(argv[2]);
            const double intercept = std::stod(argv[3]);
            std::cout << "predictions=";
            printVector({slope * std::stod(argv[4]) + intercept});
            std::cout << '\n';
            return 0;
        }
        if (operation == "nn_predict" && argc == 2) {
            std::string line;
            std::getline(std::cin, line);
            std::istringstream stream(line);
            NeuralNetwork model = NeuralNetwork::load(stream);
            const Vector x = readAndParseVectorFromStdin();
            Vector predictions;
            for (double value : x) predictions.push_back(model.predict({value})[0]);
            std::cout << "predictions=";
            printVector(predictions);
            std::cout << '\n';
            return 0;
        }
        if (!((operation == "lr_train" && argc == 2) || (operation == "nn_train_predict" && argc >= 5 && argc <= 10))) {
            throw std::invalid_argument("Invalid operation or arguments.");
        }
        const Vector x = readAndParseVectorFromStdin();
        const Vector y = readAndParseVectorFromStdin();
        if (x.empty() || x.size() != y.size()) throw std::invalid_argument("X and Y must have matching nonzero lengths.");
        const auto start = std::chrono::steady_clock::now();
        if (operation == "lr_train") {
            LinearRegression model;
            model.fit_analytical(x, y);
            std::cout << "slope=" << model.get_slope() << "\nintercept=" << model.get_intercept()
                      << "\nmse=" << model.get_mse(x, y) << "\nr_squared=" << model.get_r_squared(x, y) << '\n';
        } else {
            const auto layers = parseLayerSizes(argv[2]);
            const auto rate_values = parseVector(argv[3]);
            const auto epoch_values = parseVector(argv[4]);
            if (rate_values.size() != 1 || epoch_values.size() != 1 || epoch_values[0] < 1 ||
                epoch_values[0] > 10000 || std::floor(epoch_values[0]) != epoch_values[0]) {
                throw std::invalid_argument("Invalid training parameters.");
            }
            // Optional hidden activation, then key=value training options.
            int next = 5;
            const Activation activation =
                argc > next && std::string(argv[next]).find('=') == std::string::npos ? parse_activation(argv[next++])
                                                                                      : Activation::Sigmoid;
            TrainingOptions options;
            for (; next < argc; ++next) parseTrainingOption(argv[next], options);
            NeuralNetwork model(layers, rate_values[0], 42, activation);
            model.set_weight_decay(options.weight_decay);
            model.set_optimizer(options.optimizer, options.batch_size);
            std::vector<Vector> inputs, targets;
            for (size_t i = 0; i < x.size(); ++i) { inputs.push_back({x[i]}); targets.push_back({y[i]}); }
            const Vector evaluation_x = readAndParseVectorFromStdin();
            const Vector validation_x = readAndParseVectorFromStdin();
            const Vector validation_y = readAndParseVectorFromStdin();
            if (validation_x.size() != validation_y.size()) {
                throw std::invalid_argument("Validation X and Y must have matching lengths.");
            }
            std::vector<Vector> validation_inputs, validation_targets;
            for (size_t i = 0; i < validation_x.size(); ++i) {
                validation_inputs.push_back({validation_x[i]});
                validation_targets.push_back({validation_y[i]});
            }
            const int epochs = static_cast<int>(epoch_values[0]);
            const TrainingSummary summary = model.fit(inputs, targets, epochs, std::max(1, epochs / 100),
                                                      validation_inputs, validation_targets,
                                                      options.patience);
            const Vector& predictions = summary.predictions;
            std::cout << "nn_predictions=";
            printVector(predictions);
            Vector evaluation_predictions;
            for (double value : evaluation_x) evaluation_predictions.push_back(model.predict({value})[0]);
            std::cout << "\neval_predictions=";
            printVector(evaluation_predictions);
            std::cout << "\nepochs_run=" << summary.epochs_run;
            if (!validation_inputs.empty()) {
                std::cout << "\nbest_epoch=" << summary.best_epoch
                          << "\nstopped_early=" << (summary.stopped_early ? 1 : 0);
            }
            double mse = 0;
            for (size_t i = 0; i < y.size(); ++i) mse += std::pow(predictions[i] - y[i], 2);
            std::cout << "\nfinal_mse=" << mse / y.size() << "\nmodel=";
            model.save(std::cout);
            std::cout << '\n';
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "training_time_ms=" << ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        printUsage(argv[0]);
        return 1;
    }
}
#endif
