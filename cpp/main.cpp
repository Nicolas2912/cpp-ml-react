#include <iostream>
#include <vector>
#include <chrono>
#include "linear_regression.h"
#include "neural_network.h"

int main() {
    // Example dataset
    std::vector<double> X = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 5, 4, 5};

    // Create and train another model using analytical solution
    LinearRegression model_analytical;
    auto start_analytical = std::chrono::high_resolution_clock::now();
    model_analytical.fit_analytical(X, y);
    auto end_analytical = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_analytical = end_analytical - start_analytical;

    // Print model parameters for analytical solution
    std::cout << "Analytical Method:" << std::endl;
    std::cout << "  Slope: " << model_analytical.get_slope() << std::endl;
    std::cout << "  Intercept: " << model_analytical.get_intercept() << std::endl;
    std::cout << "  Prediction for x = 6: " << model_analytical.predict(6) << std::endl;
    std::cout << "  Training time: " << duration_analytical.count() << " ms" << std::endl;

    // --- Neural Network Example ---
    std::cout << "\nNeural Network Training:" << std::endl;

    // Create the neural network: 1 input neuron, 3 hidden neurons, 1 output neuron
    NeuralNetwork nn({1, 3, 1}, 0.05); // Added learning rate

    // Training data (same as linear regression example XOR DATA)
    std::vector<double> x_train_nn = {0.0, 1.0, 0.0, 1.0};
    std::vector<double> y_train_nn = {1.0, 0.0, 0.0, 1.0};

    // Training parameters
    int epochs = 2; // Number of times to iterate over the dataset

    std::cout << "  Training for " << epochs << " epochs..." << std::endl;
    auto start_nn = std::chrono::high_resolution_clock::now();

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < x_train_nn.size(); ++i) {
            // Prepare single input and target vectors
            Vector input_sample = {x_train_nn[i]};
            Vector target_sample = {y_train_nn[i]};

            // Train on the single sample
            nn.train(input_sample, target_sample);
        }

        // Optional: Print loss every N epochs to monitor training
        if ((epoch + 1) % 10 == 0) {
            double current_epoch_loss = 0.0;
            for (size_t i = 0; i < x_train_nn.size(); ++i) {
                Vector input_sample = {x_train_nn[i]};
                Vector predicted = nn.predict(input_sample);
                Vector target_sample = {y_train_nn[i]};
                current_epoch_loss += NeuralNetwork::mean_squared_error(predicted, target_sample);
            }
            std::cout << "  Epoch " << (epoch + 1) << "/" << epochs << ", Average Loss: " << (current_epoch_loss / x_train_nn.size()) << std::endl;
        }
    }

    auto end_nn = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_nn = end_nn - start_nn;
    std::cout << "  Training complete." << std::endl;
    std::cout << "  Training time: " << duration_nn.count() << " ms" << std::endl;


    // Test the trained network
    std::cout << "\nNeural Network Predictions:" << std::endl;
    for(double x_val : {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}) {
        Vector input_sample = {x_val};
        Vector prediction = nn.predict(input_sample);
        std::cout << "  Prediction for x = " << x_val << ": " << prediction[0] << std::endl;
    }


    return 0;
}