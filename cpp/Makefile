# Compiler
CXX = g++
# Compiler flags: -std=c++11 for modern C++, -O2 for optimization, -Wall for warnings, -fopenmp for OpenMP
CXXFLAGS = -std=c++11 -O2 -Wall -fopenmp
# Linker flags: -lm for math library, -fopenmp for OpenMP runtime
LDFLAGS = -lm -fopenmp

# Source files
SRCS = linear_regression.cpp main_server.cpp
# Object files
OBJS = $(SRCS:.cpp=.o)
# Executable name
TARGET = linear_regression_app

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile source files into object files
%.o: %.cpp linear_regression.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files - Windows compatible
clean:
	-del $(OBJS) $(TARGET) 2> nul || (exit 0) # Use del, ignore errors if files don't exist

# Phony targets
.PHONY: all clean