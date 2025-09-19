#include "linear_regression.h"
#include <iostream>
#include <ostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
#include <limits>

// Updated Constructor
LinearRegression::LinearRegression(double lr, int max_iter, int batch_size)
    : slope(0), intercept(0), learning_rate(lr), max_iterations(max_iter), batch_size(batch_size) {}

void LinearRegression::fit(const std::vector<double>& X, const std::vector<double>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y must have the same length");
    }
    if (X.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    // Pre-allocate vectors and initialize parameters
    const size_t n_samples = X.size();
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Early stopping parameters
    const double tolerance = 1e-6;
    const int patience = 5;
    double best_mse = std::numeric_limits<double>::infinity();
    int no_improvement_count = 0;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Pre-allocate batch vectors
    std::vector<double> batch_X(batch_size);
    std::vector<double> batch_y(batch_size);

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), gen);

        // Process mini-batches
        for (size_t batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
            const size_t current_batch_size = std::min(static_cast<size_t>(batch_size), n_samples - batch_start);
            
            // Prepare batch data
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < current_batch_size; ++i) {
                const size_t idx = indices[batch_start + i];
                batch_X[i] = X[idx];
                batch_y[i] = y[idx];
            }

            // Compute gradients using SIMD-friendly operations
            double slope_gradient = 0.0;
            double intercept_gradient = 0.0;

            #pragma omp parallel for reduction(+:slope_gradient, intercept_gradient) schedule(static)
            for (size_t i = 0; i < current_batch_size; ++i) {
                const double prediction = slope * batch_X[i] + intercept;
                const double error = prediction - batch_y[i];
                
                slope_gradient += error * batch_X[i];
                intercept_gradient += error;
            }

            // Update parameters
            const double batch_scale = 1.0 / current_batch_size;
            slope -= learning_rate * (slope_gradient * batch_scale);
            intercept -= learning_rate * (intercept_gradient * batch_scale);

        }

        const double epoch_mse = mean_squared_error(X, y);

        // Early stopping check
        const double improvement = best_mse - epoch_mse;
        const double relative_threshold = tolerance * std::max(1.0, best_mse);

        if (!std::isfinite(best_mse) || improvement > relative_threshold) {
            best_mse = epoch_mse;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
            if (no_improvement_count >= patience) {
                break; // Early stopping
            }
        }
    }
}

// --- Rest of the methods (predict, get_slope, get_intercept, mean, mean_squared_error) remain the same ---

double LinearRegression::predict(double x) const {
    return slope * x + intercept;
}

double LinearRegression::get_slope() const {
    return slope;
}

double LinearRegression::get_intercept() const {
    return intercept;
}

double LinearRegression::mean(const std::vector<double>& vec) const {
    if (vec.empty()) {
        throw std::invalid_argument("Cannot calculate mean of empty vector");
    }

    double sum = 0;
    // Use OpenMP for potentially large vectors, though overhead might dominate for small ones
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum / vec.size();
}

double LinearRegression::mean_squared_error(const std::vector<double>& X, const std::vector<double>& y) const {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y must have the same length");
    }
     if (X.empty()) {
        return 0.0; // Or throw an error, debatable for MSE on empty data
    }

    double mse_sum = 0;
    // Use OpenMP for parallel calculation of squared errors
    #pragma omp parallel for reduction(+:mse_sum) schedule(static)
    for (size_t i = 0; i < X.size(); ++i) {
        double prediction = slope * X[i] + intercept;
        mse_sum += std::pow(prediction - y[i], 2);
    }
    return mse_sum / X.size();
}

double LinearRegression::get_mse(const std::vector<double>& X, const std::vector<double>& y) const {
    return mean_squared_error(X, y);
}

double LinearRegression::get_r_squared(const std::vector<double>& X, const std::vector<double>& y) const {
    if (X.size() != y.size() || X.empty()) {
        throw std::invalid_argument("X and y must have the same non-empty length");
    }

    // Calculate total sum of squares (TSS)
    double y_mean = mean(y);
    double tss = 0;
    #pragma omp parallel for reduction(+:tss)
    for (size_t i = 0; i < y.size(); ++i) {
        double diff = y[i] - y_mean;
        tss += diff * diff;
    }

    // Calculate residual sum of squares (RSS)
    double rss = mean_squared_error(X, y) * X.size();

    // R² = 1 - RSS/TSS
    return 1.0 - (rss / tss);
}

void LinearRegression::fit_analytical(const std::vector<double>& X, const std::vector<double>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y must have the same length");
    }
    if (X.empty()) {
        throw std::invalid_argument("Input vectors cannot be empty");
    }

    const size_t n = X.size();
    
    // Calculate means
    const double mean_x = mean(X);
    const double mean_y = mean(y);
    
    // Calculate slope (m) using the formula: 
    // m = Σ[(x_i - mean_x)(y_i - mean_y)] / Σ[(x_i - mean_x)²]
    double numerator = 0.0;
    double denominator = 0.0;
    
    #pragma omp parallel for reduction(+:numerator, denominator)
    for (size_t i = 0; i < n; ++i) {
        double x_diff = X[i] - mean_x;
        double y_diff = y[i] - mean_y;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }
    
    // Avoid division by zero
    if (std::fabs(denominator) < 1e-10) {
        slope = 0.0;
    } else {
        slope = numerator / denominator;
    }
    
    // Calculate intercept (b) using the formula: b = mean_y - m * mean_x
    intercept = mean_y - slope * mean_x;
}
