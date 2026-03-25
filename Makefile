CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -ffast-math
SRCDIR = src
BUILDDIR = build

INCLUDES = -I$(SRCDIR)

TENSOR_ENGINE_SRC = $(SRCDIR)/tensor/cpu_engine.cpp
TENSOR_BENCHMARK = $(BUILDDIR)/tensor_benchmark

EXAMPLE_FILES := $(shell find $(SRCDIR) -name "*_example.cpp" 2>/dev/null | grep -v tensor)
BENCHMARK_FILES := $(shell find $(SRCDIR) -name "*_benchmark.cpp" 2>/dev/null | grep -v tensor)

ALGORITHMS := $(sort $(patsubst %_example,%,$(patsubst %_benchmark,%,$(basename $(notdir $(EXAMPLE_FILES) $(BENCHMARK_FILES))))))

ALGO_SET := $(filter command line environment,$(origin ALGO))

ifeq ($(ALGO),)
    ALGO := $(ALGORITHMS)
endif

VALID_ALGO := $(filter $(ALGO), $(ALGORITHMS))
INVALID_ALGO := $(filter-out $(ALGORITHMS), $(ALGO))
$(if $(INVALID_ALGO), $(warning Unknown algorithms: $(INVALID_ALGO)))

ALGO_EXAMPLES := $(addprefix $(BUILDDIR)/,$(addsuffix _example,$(VALID_ALGO)))
ALGO_BENCHMARKS := $(addprefix $(BUILDDIR)/,$(addsuffix _benchmark,$(VALID_ALGO)))

find_algo_src = $(firstword $(shell find $(SRCDIR) -name "$(1).cpp" 2>/dev/null))

$(BUILDDIR)/%_example: $(TENSOR_ENGINE_SRC)
	@# Find the example and algorithm source files
	$(eval EXAMPLE_SRC := $(call find_algo_src,$(*F)_example))
	$(eval ALGO_SRC := $(call find_algo_src,$(*F)))
	@if [ -z "$(EXAMPLE_SRC)" ]; then \
		echo "Error: Source file for $(*F)_example not found"; \
		exit 1; \
	fi
	@if [ -z "$(ALGO_SRC)" ]; then \
		echo "Error: Source file for $(*F) not found"; \
		exit 1; \
	fi
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXAMPLE_SRC) $(ALGO_SRC) $< -o $@

$(BUILDDIR)/%_benchmark: $(TENSOR_ENGINE_SRC)
	@# Find the benchmark and algorithm source files
	$(eval BENCHMARK_SRC := $(call find_algo_src,$(*F)_benchmark))
	$(eval ALGO_SRC := $(call find_algo_src,$(*F)))
	@if [ -z "$(BENCHMARK_SRC)" ]; then \
		echo "Error: Source file for $(*F)_benchmark not found"; \
		exit 1; \
	fi
	@if [ -z "$(ALGO_SRC)" ]; then \
		echo "Error: Source file for $(*F) not found"; \
		exit 1; \
	fi
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(BENCHMARK_SRC) $(ALGO_SRC) $< -o $@

.PHONY: all clean list help $(ALGORITHMS)

ifeq ($(ALGO_SET),)
all: $(TENSOR_BENCHMARK) $(ALGO_EXAMPLES) $(ALGO_BENCHMARKS)
	@echo ""
	@echo "Build complete!"
	@echo "  Built: $(notdir $(TENSOR_BENCHMARK) $(ALGO_EXAMPLES) $(ALGO_BENCHMARKS))"
else
all: $(ALGO_EXAMPLES) $(ALGO_BENCHMARKS)
	@echo ""
	@echo "Build complete!"
	@echo "  Built: $(notdir $(ALGO_EXAMPLES) $(ALGO_BENCHMARKS))"
endif

$(TENSOR_BENCHMARK): $(SRCDIR)/tensor/tensor_benchmark.cpp $(TENSOR_ENGINE_SRC)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(ALGORITHMS):
	@$(MAKE) --no-print-directory ALGO=$@

%_example:
	@$(MAKE) --no-print-directory $(BUILDDIR)/$@

%_benchmark:
	@$(MAKE) --no-print-directory $(BUILDDIR)/$@

list:
	@echo "Available algorithms:"
	@for algo in $(ALGORITHMS); do echo "  - $$algo"; done
	@if [ -z "$(ALGORITHMS)" ]; then echo "  (none found)"; fi
	@echo ""
	@echo "Usage:"
	@echo "  make                          Build all"
	@echo "  make ALGO=<name>              Build specific algorithm"
	@echo "  make <name>                   Build specific algorithm (shorthand)"
	@echo "  make <name>_example           Build only example"
	@echo "  make <name>_benchmark         Build only benchmark"

help:
	@echo "ML Library Dynamic Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)                 Build tensor_benchmark + all algorithms"
	@echo "  tensor_benchmark              Build tensor operations benchmark"
	@echo "  <algorithm>                   Build specific algorithm (example + benchmark)"
	@echo "  <algorithm>_example           Build only algorithm example"
	@echo "  <algorithm>_benchmark         Build only algorithm benchmark"
	@echo "  list                          List available algorithms"
	@echo "  help                          Show this help message"
	@echo "  clean                         Remove build directory"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Build everything"
	@echo "  make linear_regression        # Build linear regression only"
	@echo "  make ALGO=linear_regression   # Same as above"
	@echo "  make linear_regression_example  # Build only example"
	@echo ""
	@echo "Currently available algorithms:"
	@for algo in $(ALGORITHMS); do echo "  - $$algo"; done

clean:
	rm -rf $(BUILDDIR)
