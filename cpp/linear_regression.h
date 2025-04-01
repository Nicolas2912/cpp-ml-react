#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>   // Required for std::iota
#include <random>    // Required for std::shuffle, std::mt19937, std::random_device
#include <algorithm> // Required for std::min, std::shuffle
#include <omp.h>     // Required for OpenMP

class LinearRegression {
private:
    double slope;
    double intercept;
    double learning_rate;
    int max_iterations;
    int batch_size; // <-- Add batch size member

public:
    // Constructor - updated signature
    LinearRegression(double lr = 0.01, int max_iter = 1000, int batch_size = 32); // <-- Add batch size parameter

    // Train the model using gradient descent
    void fit(const std::vector<double>& X, const std::vector<double>& y);
    
    // Train the model using analytical solution (direct formula)
    void fit_analytical(const std::vector<double>& X, const std::vector<double>& y);

    // Predict using the trained model
    double predict(double x) const;

    // Getters for slope and intercept
    double get_slope() const;
    double get_intercept() const;

    // New public methods for metrics
    double get_mse(const std::vector<double>& X, const std::vector<double>& y) const;
    double get_r_squared(const std::vector<double>& X, const std::vector<double>& y) const;

private:
    // Calculate mean of a vector
    double mean(const std::vector<double>& vec) const;

    // Calculate mean squared error
    double mean_squared_error(const std::vector<double>& X, const std::vector<double>& y) const;
};

#endif // LINEAR_REGRESSION_H