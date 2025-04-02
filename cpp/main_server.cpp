// --- START OF FILE main_server.cpp ---

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "linear_regression.h"
#include "neural_network.h"

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// Helper function parseVector (no changes)
std::vector<double> parseVector(const std::string& s) {
    // ... (keep existing implementation) ...
    std::vector<double> result;
    if (s.empty()) {
        return result;
    }
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            char* end;
            double val = std::strtod(item.c_str(), &end);
            if (*end != '\0' || std::isinf(val) || std::isnan(val)) {
                 throw std::invalid_argument("Invalid numeric value in input: '" + item + "'");
            }
            result.push_back(val);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Error parsing value: Invalid argument '" << item << "'" << std::endl;
            throw;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Error parsing value: Out of range '" << item << "'" << std::endl;
            throw;
        }
    }
    return result;
}

// Helper function parseLayerSizes (no changes)
std::vector<size_t> parseLayerSizes(const std::string& s) {
    // ... (keep existing implementation) ...
     std::vector<size_t> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, '-')) {
         if (item.empty()) continue;
         try {
            char* end;
            unsigned long val_ul = std::strtoul(item.c_str(), &end, 10);
             if (*end != '\0' || val_ul > std::numeric_limits<size_t>::max() || (val_ul == 0 && item != "0")) {
                 throw std::invalid_argument("Invalid layer size value: '" + item + "'");
             }
             size_t val = static_cast<size_t>(val_ul);
             if (val == 0) {
                 throw std::invalid_argument("Layer size cannot be zero: '" + item + "'");
             }
             result.push_back(val);
         } catch (const std::invalid_argument& ia) {
            std::cerr << "Error parsing layer size: Invalid argument '" << item << "'" << std::endl;
            throw;
         } catch (const std::out_of_range& oor) {
            std::cerr << "Error parsing layer size: Out of range '" << item << "'" << std::endl;
            throw;
         }
    }
     if (result.size() < 2) {
        throw std::invalid_argument("Invalid layer structure: Must have at least an input and output layer (e.g., '1-1'). Received: '" + s + "'");
    }
    return result;
}

// Helper function readAndParseVectorFromStdin (no changes)
std::vector<double> readAndParseVectorFromStdin() {
    // ... (keep existing implementation) ...
     std::string line;
    if (std::getline(std::cin, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
        return parseVector(line);
    } else {
        if (std::cin.eof()) {
            // EOF is okay
        } else if (std::cin.fail()) {
            std::cerr << "Error: Failed to read data line from standard input." << std::endl;
        }
        return {};
    }
}

// Helper to print a vector (no changes)
void printVector(const Vector& vec) {
     // ... (keep existing implementation) ...
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ",");
    }
}

// Updated usage message function (no changes)
void printUsage(const char* progName) {
    // ... (keep existing implementation) ...
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << progName << " lr_train" << std::endl;
    std::cerr << "    (Reads X and Y from stdin, 1 line each, comma-separated)" << std::endl;
    std::cerr << "  " << progName << " lr_predict <slope> <intercept> <x_value>" << std::endl;
    std::cerr << "  " << progName << " nn_train_predict <layers> <learning_rate> <epochs>" << std::endl; // Kept command name
    std::cerr << "    (e.g., " << progName << " nn_train_predict 1-5-1 0.05 1000)" << std::endl;
    std::cerr << "    (Reads X and Y from stdin, 1 line each, comma-separated)" << std::endl;
    std::cerr << "    (Trains NN using train_for_epochs, outputs loss updates and final predictions)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout.precision(std::numeric_limits<double>::max_digits10);

    if (argc < 2) {
        std::cerr << "Error: Operation mode required." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::string operation = argv[1];

    try {
        // --- Linear Regression Training Mode --- (No changes needed)
        if (operation == "lr_train") {
            // ... (keep existing implementation) ...
             if (argc != 2) { /* ... */ return 1; }
             std::vector<double> X = readAndParseVectorFromStdin();
             std::vector<double> y = readAndParseVectorFromStdin();
             if (X.empty() || y.empty()) { /* ... */ return 1; }
             if (X.size() != y.size()) { /* ... */ return 1; }
             LinearRegression model;
             auto start_time = std::chrono::high_resolution_clock::now();
             model.fit_analytical(X, y);
             auto end_time = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
             std::cout << "slope=" << model.get_slope() << std::endl;
             std::cout << "intercept=" << model.get_intercept() << std::endl;
             std::cout << "training_time_ms=" << duration.count() << std::endl;
             std::cout << "mse=" << model.get_mse(X, y) << std::endl;
             std::cout << "r_squared=" << model.get_r_squared(X, y) << std::endl;

        // --- Linear Regression Prediction Mode --- (No changes needed)
        } else if (operation == "lr_predict") {
             // ... (keep existing implementation) ...
             if (argc != 5) { /* ... */ return 1; }
             double slope = std::stod(argv[2]);
             double intercept = std::stod(argv[3]);
             double x_value = std::stod(argv[4]);
             double prediction = slope * x_value + intercept;
             std::cout << "prediction=" << prediction << std::endl;

        // --- Neural Network Training & Prediction Mode (MODIFIED) ---
        } else if (operation == "nn_train_predict") { // Keep command name consistent
            if (argc != 5) {
                std::cerr << "Error: Invalid arguments for operation '" << operation << "'." << std::endl;
                printUsage(argv[0]);
                return 1;
            }

            // Parse NN parameters (same as before)
            std::vector<size_t> layer_sizes = parseLayerSizes(argv[2]);
            double learning_rate = std::stod(argv[3]);
            int epochs = std::stoi(argv[4]);

            // Validation (same as before)
            if (epochs <= 0) { /* ... */ return 1; }
            if (learning_rate <= 0) { /* ... warning ... */ }

            // Read X and y values from stdin (same as before)
            std::vector<double> X_train_flat = readAndParseVectorFromStdin();
            std::vector<double> y_train_flat = readAndParseVectorFromStdin();

             // Validation (same as before)
            if (X_train_flat.empty() || y_train_flat.empty()) { /* ... */ return 1; }
            if (X_train_flat.size() != y_train_flat.size()) { /* ... */ return 1; }
            if (layer_sizes[0] != 1) { /* ... */ return 1; }
            if (layer_sizes.back() != 1) { /* ... */ return 1; }

            // --- MODIFICATION START ---
            // Convert flat vectors to vector<Vector> format for train_for_epochs
            std::vector<Vector> X_train_vec;
            std::vector<Vector> y_train_vec;
            X_train_vec.reserve(X_train_flat.size());
            y_train_vec.reserve(y_train_flat.size());
            for(size_t i = 0; i < X_train_flat.size(); ++i) {
                X_train_vec.push_back({X_train_flat[i]});
                y_train_vec.push_back({y_train_flat[i]});
            }
            // --- MODIFICATION END ---


            // Create the neural network
            NeuralNetwork nn(layer_sizes, learning_rate);

            auto start_time = std::chrono::high_resolution_clock::now();

            // --- MODIFICATION START ---
            // Call train_for_epochs instead of manual loop
            // This function will print loss updates to stdout periodically
            // It assumes report_every_n_epochs defaults to 10 or another value inside the class
            Vector final_predictions_flat = nn.train_for_epochs(X_train_vec, y_train_vec, epochs);
            // --- MODIFICATION END ---

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // --- MODIFICATION START ---
            // Calculate final MSE AFTER training using the returned predictions
             double final_mse = 0.0;
             if (final_predictions_flat.size() == y_train_flat.size()) {
                 for (size_t i = 0; i < final_predictions_flat.size(); ++i) {
                     // Note: y_train_vec[i][0] is equivalent to y_train_flat[i]
                     double error = final_predictions_flat[i] - y_train_flat[i];
                     final_mse += error * error;
                 }
                 final_mse /= final_predictions_flat.size();
             } else {
                 final_mse = std::numeric_limits<double>::quiet_NaN(); // Indicate error
                 std::cerr << "Warning: Prediction vector size mismatch after training." << std::endl;
             }


            // Output final results AFTER training is complete
            // Loss updates were already printed during the train_for_epochs call
            std::cout << "training_time_ms=" << duration.count() << std::endl;
            std::cout << "final_mse=" << final_mse << std::endl; // Use the calculated final MSE
            std::cout << "nn_predictions=";
            printVector(final_predictions_flat); // Use the predictions returned by train_for_epochs
            std::cout << std::endl;
            // --- MODIFICATION END ---

        } else {
            std::cerr << "Error: Unknown operation '" << operation << "'." << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Input Error: " << e.what() << std::endl;
        printUsage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
        return 1;
    }

    return 0;
}
// --- END OF FILE main_server.cpp ---