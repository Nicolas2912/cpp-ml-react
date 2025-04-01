// --- START OF FILE main_server.cpp ---

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <chrono> // Include for timing
#include <numeric> // For std::iota, std::inner_product
#include <random>  // For std::shuffle, std::mt19937
#include <algorithm> // For std::min, std::shuffle, std::transform
#include <cmath> // For std::sqrt, std::exp
#include <cstdlib> // For std::strtod, std::strtoul

// Include both model headers
#include "linear_regression.h"
#include "neural_network.h" // <<< ADDED

// Define aliases for clarity (assuming these were in neural_network.h or similar)
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;


// Helper function to parse a comma-separated string into a vector of doubles
std::vector<double> parseVector(const std::string& s) {
    std::vector<double> result;
    if (s.empty()) {
        return result; // Return empty vector if input string is empty
    }
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            // Use strtod for potentially better error handling edge cases
            char* end;
            double val = std::strtod(item.c_str(), &end);
            // Check if the whole string was consumed and it's not infinity/NaN
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

// Helper function to parse layer sizes string (e.g., "1-3-1")
std::vector<size_t> parseLayerSizes(const std::string& s) {
    std::vector<size_t> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, '-')) {
         if (item.empty()) continue; // Skip empty parts (e.g., trailing '-')
         try {
            // Use strtoul for unsigned long (size_t)
            char* end;
            unsigned long val_ul = std::strtoul(item.c_str(), &end, 10);
            // Check if conversion was successful and value is within size_t limits
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


// Helper function to read a line from stdin and parse it into a vector
std::vector<double> readAndParseVectorFromStdin() {
    std::string line;
    if (std::getline(std::cin, line)) {
        // Trim leading/trailing whitespace which might interfere
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);
        return parseVector(line);
    } else {
        // Handle potential error or end-of-input during reading
        if (std::cin.eof()) {
            // Don't print error on expected EOF if reading multiple lines,
            // but return empty to signal failure.
        } else if (std::cin.fail()) {
            std::cerr << "Error: Failed to read data line from standard input." << std::endl;
        }
        return {}; // Return empty vector on error/eof
    }
}

// Helper to print a vector as comma-separated values
void printVector(const Vector& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ",");
    }
}

// Updated usage message function
void printUsage(const char* progName) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << progName << " lr_train" << std::endl;
    std::cerr << "    (Reads X and Y from stdin, 1 line each, comma-separated)" << std::endl;
    std::cerr << "  " << progName << " lr_predict <slope> <intercept> <x_value>" << std::endl;
    std::cerr << "  " << progName << " nn_train_predict <layers> <learning_rate> <epochs>" << std::endl;
    std::cerr << "    (e.g., " << progName << " nn_train_predict 1-5-1 0.05 1000)" << std::endl;
    std::cerr << "    (Reads X and Y from stdin, 1 line each, comma-separated)" << std::endl;
    std::cerr << "    (Trains NN and outputs predictions for the input X values)" << std::endl;
}

int main(int argc, char* argv[]) {
    // Set high precision for output
    std::cout.precision(std::numeric_limits<double>::max_digits10);

    if (argc < 2) {
        std::cerr << "Error: Operation mode required." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::string operation = argv[1];

    try {
        // --- Linear Regression Training Mode ---
        if (operation == "lr_train") {
             if (argc != 2) {
                 std::cerr << "Error: Invalid arguments for operation '" << operation << "'." << std::endl;
                 printUsage(argv[0]);
                 return 1;
             }
            // Read X and y values from stdin (expects two lines)
            std::vector<double> X = readAndParseVectorFromStdin();
            std::vector<double> y = readAndParseVectorFromStdin();

            if (X.empty() || y.empty()) {
                 std::cerr << "Error: Failed to read valid X and Y data from standard input." << std::endl;
                 return 1;
            }
            if (X.size() != y.size()) {
                std::cerr << "Error: X and y must have the same number of elements." << std::endl;
                return 1;
            }

            LinearRegression model;
            auto start_time = std::chrono::high_resolution_clock::now();
            model.fit_analytical(X, y); // Use analytical for LR server
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "slope=" << model.get_slope() << std::endl;
            std::cout << "intercept=" << model.get_intercept() << std::endl;
            std::cout << "training_time_ms=" << duration.count() << std::endl;
            std::cout << "mse=" << model.get_mse(X, y) << std::endl;
            std::cout << "r_squared=" << model.get_r_squared(X, y) << std::endl;

        // --- Linear Regression Prediction Mode ---
        } else if (operation == "lr_predict") {
             if (argc != 5) {
                 std::cerr << "Error: Invalid arguments for operation '" << operation << "'." << std::endl;
                 printUsage(argv[0]);
                 return 1;
             }
             double slope = std::stod(argv[2]);
             double intercept = std::stod(argv[3]);
             double x_value = std::stod(argv[4]);
             // Use a temporary model instance just for the predict method if needed
             // Or just calculate directly if predict() is simple
             // LinearRegression temp_model; temp_model.set_slope(slope); temp_model.set_intercept(intercept);
             // double prediction = temp_model.predict(x_value);
             double prediction = slope * x_value + intercept; // Direct calculation
             std::cout << "prediction=" << prediction << std::endl;

        // --- Neural Network Training & Prediction Mode ---
        } else if (operation == "nn_train_predict") {
            if (argc != 5) {
                std::cerr << "Error: Invalid arguments for operation '" << operation << "'." << std::endl;
                printUsage(argv[0]);
                return 1;
            }

            // Parse NN parameters
            std::vector<size_t> layer_sizes = parseLayerSizes(argv[2]);
            double learning_rate = std::stod(argv[3]);
            int epochs = std::stoi(argv[4]); // Use stoi for integer

            if (epochs <= 0) {
                 std::cerr << "Error: Epochs must be a positive integer." << std::endl;
                 return 1;
            }
             if (learning_rate <= 0) {
                 std::cerr << "Warning: Learning rate is non-positive (" << learning_rate << ")." << std::endl;
                 // Allow zero/negative learning rate, although it might not train.
             }


            // Read X and y values from stdin (expects two lines)
            std::vector<double> X_train = readAndParseVectorFromStdin();
            std::vector<double> y_train = readAndParseVectorFromStdin();

            if (X_train.empty() || y_train.empty()) {
                 std::cerr << "Error: Failed to read valid X and Y training data from standard input." << std::endl;
                 return 1;
            }
            if (X_train.size() != y_train.size()) {
                std::cerr << "Error: X and Y training data must have the same number of elements." << std::endl;
                return 1;
            }
            if (layer_sizes[0] != 1) {
                 std::cerr << "Error: For this simple integration, the input layer size must be 1 (matching 1D input data)." << std::endl;
                 return 1;
            }
             if (layer_sizes.back() != 1) {
                 std::cerr << "Error: For this simple integration, the output layer size must be 1 (matching 1D target data)." << std::endl;
                 return 1;
            }

            // Create and train the neural network
            NeuralNetwork nn(layer_sizes, learning_rate);

            auto start_time = std::chrono::high_resolution_clock::now();

            // Training loop
            for (int epoch = 0; epoch < epochs; ++epoch) {
                // Here we implement SGD - update after each sample
                for (size_t i = 0; i < X_train.size(); ++i) {
                    Vector input_sample = {X_train[i]};
                    Vector target_sample = {y_train[i]};
                    nn.train(input_sample, target_sample); // Uses backprop internally
                }
                // Optional: Add logic here to print progress every N epochs to stderr if desired
                // if ((epoch + 1) % 100 == 0) { std::cerr << "Epoch " << (epoch+1) << "/" << epochs << std::endl;}
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // Calculate final loss and predictions after training
            double final_mse = 0.0;
            Vector final_predictions(X_train.size());
            for (size_t i = 0; i < X_train.size(); ++i) {
                Vector input_sample = {X_train[i]};
                Vector predicted_sample = nn.predict(input_sample);
                Vector target_sample = {y_train[i]};
                final_predictions[i] = predicted_sample[0]; // Store the prediction
                // Use the static MSE method if available, otherwise recalculate
                double sample_error = predicted_sample[0] - target_sample[0];
                final_mse += sample_error * sample_error;
            }
            final_mse /= X_train.size(); // Average the MSE

            // Output results
            std::cout << "training_time_ms=" << duration.count() << std::endl;
            std::cout << "final_mse=" << final_mse << std::endl;
            std::cout << "nn_predictions=";
            printVector(final_predictions);
            std::cout << std::endl;

        } else {
            std::cerr << "Error: Unknown operation '" << operation << "'." << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Input Error: " << e.what() << std::endl;
        // Consider printing usage here too
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