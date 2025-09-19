# Compiler (can be overridden: `make CXX=clang++`)
CXX ?= g++

# Detect whether the selected compiler is clang; Apple's clang front-end reports
# itself as both `clang` and `clang++`. In that case, OpenMP needs different flags.
CXX_VERSION := $(shell $(CXX) --version 2>/dev/null)
USING_CLANG := $(findstring clang,$(CXX_VERSION))

# Default OpenMP flags assume GCC (or LLVM's `g++` front-end) support.
OPENMP_CFLAGS ?= -fopenmp
OPENMP_LDFLAGS ?= -fopenmp

# Apple's clang requires the OpenMP runtime (`libomp`) and the `-Xpreprocessor`
# shim to enable the OpenMP pragma handling. We try to discover Homebrew's libomp
# installation and fall back to a sequential build otherwise.
ifneq ($(USING_CLANG),)
  LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
  ifneq ($(LIBOMP_PREFIX),)
    OPENMP_CFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    OPENMP_LDFLAGS := -L$(LIBOMP_PREFIX)/lib -lomp
  else
    $(warning OpenMP runtime (libomp) not detected; building without OpenMP. Install via `brew install libomp` to enable parallel execution.)
    OPENMP_CFLAGS :=
    OPENMP_LDFLAGS :=
  endif
endif

# Compiler flags: -std=c++11 for modern C++, -O2 for optimization, -Wall for warnings.
CXXFLAGS = -std=c++11 -O2 -Wall $(OPENMP_CFLAGS)
# Linker flags: -lm for math library, include OpenMP runtime when needed.
LDFLAGS = -lm $(OPENMP_LDFLAGS)

# Source files
SRCS = linear_regression.cpp main_server.cpp neural_network.cpp
# Object files
OBJS = $(SRCS:.cpp=.o)
# Executable name
TARGET = linear_regression_app

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Compile source files into object files
%.o: %.cpp linear_regression.h neural_network.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files - Windows compatible
clean:
	rm -f $(OBJS) $(TARGET) $(TEST_TARGETS)

# Test targets
TEST_TARGETS = linear_regression_tests neural_network_tests main_server_tests

linear_regression_tests: tests/linear_regression_tests.cpp linear_regression.cpp linear_regression.h
	$(CXX) $(CXXFLAGS) tests/linear_regression_tests.cpp linear_regression.cpp -o $@ $(LDFLAGS)

neural_network_tests: tests/neural_network_tests.cpp neural_network.cpp neural_network.h
	$(CXX) $(CXXFLAGS) tests/neural_network_tests.cpp neural_network.cpp -o $@ $(LDFLAGS)

main_server_tests: tests/main_server_tests.cpp linear_regression.cpp neural_network.cpp
	$(CXX) $(CXXFLAGS) -DUNIT_TESTING tests/main_server_tests.cpp linear_regression.cpp neural_network.cpp -o $@ $(LDFLAGS)

tests: $(TEST_TARGETS)

test_all: tests
	./linear_regression_tests
	./neural_network_tests
	./main_server_tests

coverage: clean
	$(MAKE) CXXFLAGS="$(CXXFLAGS) --coverage -O0" LDFLAGS="$(LDFLAGS) --coverage" tests
	./linear_regression_tests
	./neural_network_tests
	./main_server_tests
	gcov -o . linear_regression.cpp neural_network.cpp main_server.cpp

# Phony targets
.PHONY: all clean tests test_all coverage $(TEST_TARGETS)
