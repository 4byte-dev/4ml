CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -ffast-math
SRCDIR = src
BUILDDIR = build

INCLUDES = -I$(SRCDIR)

# Source files
TENSOR_ENGINE_SRC = $(SRCDIR)/tensor/cpu_engine.cpp
LR_SRC = $(SRCDIR)/supervised/regression/linear_regression.cpp

# Targets
TENSOR_BENCHMARK = $(BUILDDIR)/tensor_benchmark
LR_EXAMPLE = $(BUILDDIR)/linear_regression_example
LR_BENCHMARK = $(BUILDDIR)/linear_regression_benchmark

.PHONY: all clean

all: $(TENSOR_BENCHMARK) $(LR_EXAMPLE) $(LR_BENCHMARK)

$(TENSOR_BENCHMARK): $(SRCDIR)/tensor/tensor_benchmark.cpp $(TENSOR_ENGINE_SRC)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(LR_EXAMPLE): $(SRCDIR)/supervised/regression/linear_regression_example.cpp $(LR_SRC) $(TENSOR_ENGINE_SRC)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(LR_BENCHMARK): $(SRCDIR)/supervised/regression/linear_regression_benchmark.cpp $(LR_SRC) $(TENSOR_ENGINE_SRC)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -rf $(BUILDDIR)
