#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../main_server.cpp"

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

    template <typename Func>
    void expectThrows(const std::string& name, Func&& func) {
        ++total;
        bool threw_expected = false;
        std::ostringstream sink;
        auto* original_buf = std::cerr.rdbuf(sink.rdbuf());
        try {
            func();
        } catch (const std::invalid_argument&) {
            threw_expected = true;
        } catch (const std::exception& e) {
            std::cerr.rdbuf(original_buf);
            ++failed;
            std::cerr << "[FAIL] " << name << ": threw unexpected std::exception: " << e.what() << std::endl;
            return;
        } catch (...) {
            std::cerr.rdbuf(original_buf);
            ++failed;
            std::cerr << "[FAIL] " << name << ": threw unexpected non-std exception" << std::endl;
            return;
        }

        std::cerr.rdbuf(original_buf);
        if (threw_expected) {
            std::cout << "[PASS] " << name << std::endl;
        } else {
            ++failed;
            std::cerr << "[FAIL] " << name << ": expected std::invalid_argument" << std::endl;
        }
    }
};

class StreamRedirect {
public:
    StreamRedirect(std::ostream& stream, std::ostream& sink)
        : stream_(stream), original_(stream.rdbuf(sink.rdbuf())) {}
    ~StreamRedirect() { stream_.rdbuf(original_); }
private:
    std::ostream& stream_;
    std::streambuf* original_;
};

} // namespace

int main() {
    TestRunner runner;

    {
        auto parsed = parseVector("1,2,-3.5");
        runner.expectTrue(parsed.size() == 3 && std::fabs(parsed[2] + 3.5) < 1e-12,
                          "parseVector parses comma separated doubles");
    }

    runner.expectThrows("parseVector rejects invalid token", [] {
        parseVector("1, x");
    });

    runner.expectThrows("parseVector rejects values with trailing spaces", [] {
        parseVector("1, 2 ");
    });

    {
        auto sizes = parseLayerSizes("1-3-1");
        runner.expectTrue(sizes.size() == 3 && sizes[1] == 3,
                          "parseLayerSizes parses dash separated sizes");
    }

    runner.expectThrows("parseLayerSizes rejects zeros", [] {
        parseLayerSizes("1-0-1");
    });

    {
        std::istringstream input_stream("1.0,2.5\n");
        auto* original_buf = std::cin.rdbuf(input_stream.rdbuf());
        std::vector<double> parsed = readAndParseVectorFromStdin();
        std::cin.rdbuf(original_buf);
        runner.expectTrue(parsed.size() == 2 && std::fabs(parsed[1] - 2.5) < 1e-12,
                          "readAndParseVectorFromStdin returns parsed values");
    }

    runner.expectThrows("readAndParseVectorFromStdin propagates parse errors", [] {
        std::istringstream input_stream("1.0 , 2.5\n");
        auto* original_buf = std::cin.rdbuf(input_stream.rdbuf());
        try {
            (void)readAndParseVectorFromStdin();
        } catch (...) {
            std::cin.rdbuf(original_buf);
            throw;
        }
        std::cin.rdbuf(original_buf);
    });

    {
        std::ostringstream capture;
        {
            StreamRedirect redirect(std::cout, capture);
            printVector({1.0, 2.0, 3.0});
        }
        runner.expectTrue(capture.str() == "1,2,3",
                          "printVector joins entries with commas");
    }

    {
        std::ostringstream capture;
        StreamRedirect redirect(std::cerr, capture);
        printUsage("app");
        runner.expectTrue(capture.str().find("Usage:") != std::string::npos,
                          "printUsage prints usage header");
    }

    if (runner.failed == 0) {
        std::cout << "\nAll " << runner.total << " CLI helper tests passed." << std::endl;
        return 0;
    }

    std::cerr << "\n" << runner.failed << " of " << runner.total << " CLI helper tests failed." << std::endl;
    return 1;
}
