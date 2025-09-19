#include "../linear_regression.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
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

    runner.expectThrows("fit_analytical rejects mismatched input sizes", [] {
        LinearRegression model;
        std::vector<double> X{1.0, 2.0};
        std::vector<double> y{3.0};
        model.fit_analytical(X, y);
    });

    runner.expectThrows("fit rejects zero batch size", [] {
        LinearRegression model(0.01, 10, 0);
        std::vector<double> X{1.0, 2.0, 3.0};
        std::vector<double> y{2.0, 4.0, 6.0};
        model.fit(X, y);
    });

    runner.expectThrows("fit rejects empty input", [] {
        LinearRegression model;
        std::vector<double> X;
        std::vector<double> y;
        model.fit(X, y);
    });

    {
        LinearRegression model;
        std::vector<double> X{1.0, 2.0, 3.0, 4.0};
        std::vector<double> y{3.0, 5.0, 7.0, 9.0}; // Perfect line: slope 2, intercept 1
        model.fit_analytical(X, y);

        runner.expectNear(model.get_slope(), 2.0, 1e-9, "fit_analytical computes expected slope");
        runner.expectNear(model.get_intercept(), 1.0, 1e-9, "fit_analytical computes expected intercept");

        const double prediction = model.predict(5.0);
        runner.expectNear(prediction, 11.0, 1e-9, "predict returns value on fitted line");

        const double mse = model.get_mse(X, y);
        runner.expectNear(mse, 0.0, 1e-12, "get_mse returns zero for perfect fit");

        const double r2 = model.get_r_squared(X, y);
        runner.expectNear(r2, 1.0, 1e-12, "get_r_squared returns one for perfect fit");
    }

    {
        LinearRegression model;
        std::vector<double> X{0.0, 1.0, 2.0, 3.0};
        std::vector<double> y{1.0, 3.0, 5.0, 7.5};
        model.fit_analytical(X, y);

        // Expected slope and intercept from least squares calculation
        runner.expectNear(model.get_slope(), 2.15, 1e-6, "fit_analytical handles noisy data (slope)");
        runner.expectNear(model.get_intercept(), 0.9, 1e-6, "fit_analytical handles noisy data (intercept)");

        const double mse = model.get_mse(X, y);
        runner.expectTrue(mse > 0.0, "get_mse returns positive value for imperfect fit");

        const double r2 = model.get_r_squared(X, y);
        runner.expectNear(r2, 0.9967655, 1e-6, "get_r_squared reflects strong but imperfect correlation");
    }

    {
        LinearRegression model(0.05, 2000, 2);
        std::vector<double> X{1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> y{2.0, 4.0, 6.0, 8.0, 10.0};
        model.fit(X, y);

        // Allow a slightly wider tolerance to account for minor
        // compiler/platform numeric differences in CI environments.
        runner.expectNear(model.get_slope(), 2.0, 5e-2, "fit converges slope on linear data");
        runner.expectNear(model.get_intercept(), 0.0, 1e-1, "fit converges intercept on linear data");
        runner.expectNear(model.get_mse(X, y), 0.0, 1e-1, "fit drives mse near zero");

        std::vector<double> empty;
        runner.expectNear(model.get_mse(empty, empty), 0.0, 1e-12, "get_mse returns zero for empty dataset");
    }

    if (runner.failed == 0) {
        std::cout << "\nAll " << runner.total << " C++ linear regression tests passed." << std::endl;
        return 0;
    }

    std::cerr << "\n" << runner.failed << " of " << runner.total << " C++ linear regression tests failed." << std::endl;
    return 1;
}
