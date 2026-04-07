CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -ffast-math
SRCDIR = src
BUILDDIR = build

INCLUDES = -I$(SRCDIR)

TENSOR_ENGINE_SRC = $(SRCDIR)/tensor/cpu_engine.cpp
TENSOR_BENCHMARK = $(BUILDDIR)/tensor_benchmark

ALGO_SRCS := $(shell find $(SRCDIR)/supervised -path "*/linear_square_error/*.cpp" ! -name "*_test.cpp" ! -name "*_example.cpp" ! -name "*_benchmark.cpp" 2>/dev/null)
ALGO_NAMES := $(sort $(basename $(notdir $(ALGO_SRCS))))

ALGO_SET := $(filter command line environment,$(origin ALGO))

ifeq ($(ALGO),)
    ALGO := $(ALGO_NAMES)
endif

VALID_ALGO := $(filter $(ALGO), $(ALGO_NAMES))
INVALID_ALGO := $(filter-out $(ALGO), $(ALGO_NAMES))
$(if $(INVALID_ALGO), $(warning Unknown algorithms: $(INVALID_ALGO)))

ALGO_EXAMPLES := $(addprefix $(BUILDDIR)/,$(addsuffix _example,$(VALID_ALGO)))
ALGO_BENCHMARKS := $(addprefix $(BUILDDIR)/,$(addsuffix _benchmark,$(VALID_ALGO)))
ALGO_TESTS := $(addprefix $(BUILDDIR)/,$(addsuffix _test,$(VALID_ALGO)))

find_algo_src = $(shell find $(SRCDIR)/supervised -name "$(1).cpp" 2>/dev/null | head -1)

TEST_FILES := $(shell find $(SRCDIR)/supervised -name "*_test.cpp" 2>/dev/null)
TEST_BINS := $(ALGO_TESTS)

GTEST_CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I$(SRCDIR) -I/usr/local/include
GTEST_LDFLAGS = -L/usr/local/lib -lgtest -lgtest_main -lpthread

.PHONY: all clean list help test $(ALGO_NAMES)

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

$(BUILDDIR)/%_example: $(TENSOR_ENGINE_SRC)
	$(eval algo := $(*F))
	$(eval EXAMPLE_SRC := $(shell find $(SRCDIR)/supervised -name "$(algo)_example.cpp" 2>/dev/null | head -1))
	$(eval ALGO_SRC := $(shell find $(SRCDIR)/supervised -name "$(algo).cpp" 2>/dev/null | grep -v example | grep -v benchmark | grep -v test | head -1))
	@if [ -z "$(EXAMPLE_SRC)" ]; then \
		echo "Error: Example source for $(algo) not found"; exit 1; fi
	@if [ -z "$(ALGO_SRC)" ]; then \
		echo "Error: Algorithm source for $(algo) not found"; exit 1; fi
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXAMPLE_SRC) $(ALGO_SRC) $(TENSOR_ENGINE_SRC) -o $@

$(BUILDDIR)/%_benchmark: $(TENSOR_ENGINE_SRC)
	$(eval algo := $(*F))
	$(eval BENCHMARK_SRC := $(shell find $(SRCDIR)/supervised -name "$(algo)_benchmark.cpp" 2>/dev/null | head -1))
	$(eval ALGO_SRC := $(shell find $(SRCDIR)/supervised -name "$(algo).cpp" 2>/dev/null | grep -v example | grep -v benchmark | grep -v test | head -1))
	@if [ -z "$(BENCHMARK_SRC)" ]; then \
		echo "Error: Benchmark source for $(algo) not found"; exit 1; fi
	@if [ -z "$(ALGO_SRC)" ]; then \
		echo "Error: Algorithm source for $(algo) not found"; exit 1; fi
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(BENCHMARK_SRC) $(ALGO_SRC) $(TENSOR_ENGINE_SRC) -o $@

$(BUILDDIR)/%_test: $(TENSOR_ENGINE_SRC)
	$(eval algo := $(*F))
	$(eval TEST_SRC := $(shell find $(SRCDIR)/supervised -name "$(algo)_test.cpp" 2>/dev/null | head -1))
	$(eval ALGO_SRC := $(shell find $(SRCDIR)/supervised -name "$(algo).cpp" 2>/dev/null | grep -v example | grep -v benchmark | grep -v test | head -1))
	@if [ -z "$(TEST_SRC)" ]; then \
		echo "Error: Test source for $(algo) not found"; exit 1; fi
	@if [ -z "$(ALGO_SRC)" ]; then \
		echo "Error: Algorithm source for $(algo) not found"; exit 1; fi
	@mkdir -p $(BUILDDIR)
	$(CXX) $(GTEST_CXXFLAGS) $(TEST_SRC) $(ALGO_SRC) $(TENSOR_ENGINE_SRC) $(GTEST_LDFLAGS) -o $@

$(ALGO_NAMES):
	@$(MAKE) --no-print-directory ALGO=$@

$(foreach algo,$(ALGO_NAMES),\
  $(eval .PHONY: $(algo)_test)\
  $(eval $(algo)_test: ; @$$(MAKE) --no-print-directory $$(BUILDDIR)/$(algo)_test)\
)

$(foreach algo,$(ALGO_NAMES),\
  $(eval .PHONY: $(algo)_example)\
  $(eval $(algo)_example: ; @$$(MAKE) --no-print-directory $$(BUILDDIR)/$(algo)_example)\
)

$(foreach algo,$(ALGO_NAMES),\
  $(eval .PHONY: $(algo)_benchmark)\
  $(eval $(algo)_benchmark: ; @$$(MAKE) --no-print-directory $$(BUILDDIR)/$(algo)_benchmark)\
)

test: $(TEST_BINS)
	@echo ""
	@echo "Running tests..."
	@failed=0; total=0; \
	for bin in $(TEST_BINS); do \
		total=$$((total + 1)); \
		echo "--- $$bin ---"; \
		if $$bin; then \
			echo "PASSED"; \
		else \
			echo "FAILED"; \
			failed=$$((failed + 1)); \
		fi; \
		echo ""; \
	done; \
	echo "========================================"; \
	echo "Results: $$((total - failed))/$$total passed"; \
	if [ $$failed -gt 0 ]; then \
		echo "$$failed test suite(s) FAILED"; \
		exit 1; \
	else \
		echo "All test suites passed!"; \
	fi

ifeq ($(ALGO_SET),)
else
override test: TEST_BINS := $(ALGO_TESTS)
endif

list:
	@echo "Available algorithms:"
	@for algo in $(ALGO_NAMES); do echo "  - $$algo"; done
	@if [ -z "$(ALGO_NAMES)" ]; then echo "  (none found)"; fi
	@echo ""
	@echo "Available tests:"
	@for f in $(notdir $(TEST_FILES)); do echo "  - $$(basename $$f .cpp)"; done
	@if [ -z "$(TEST_FILES)" ]; then echo "  (none found)"; fi
	@echo ""
	@echo "Usage:"
	@echo "  make                          Build all"
	@echo "  make ALGO=<name>              Build specific algorithm"
	@echo "  make <name>                   Build specific algorithm (shorthand)"
	@echo "  make <name>_example           Build only example"
	@echo "  make <name>_benchmark         Build only benchmark"
	@echo "  make <name>_test              Build and run specific test"

help:
	@echo "ML Library Dynamic Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)                 Build tensor_benchmark + all algorithms"
	@echo "  test                          Build and run all *_test.cpp files"
	@echo "  test ALGO=<name>              Run tests for specific algorithm only"
	@echo "  <algorithm>_test              Build and run specific algorithm test"
	@echo "  tensor_benchmark              Build tensor operations benchmark"
	@echo "  <algorithm>                   Build specific algorithm (example + benchmark)"
	@echo "  <algorithm>_example           Build only algorithm example"
	@echo "  <algorithm>_benchmark         Build only algorithm benchmark"
	@echo "  list                          List available algorithms and tests"
	@echo "  help                          Show this help message"
	@echo "  clean                         Remove build directory"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Build everything"
	@echo "  make test                     # Build and run all tests"
	@echo "  make test ALGO=linear_square_error  # Run linear_square_error tests only"
	@echo "  make linear_square_error_test   # Build and run linear_square_error test"
	@echo "  make linear_square_error        # Build linear square error only"
	@echo "  make ALGO=linear_square_error   # Same as above"
	@echo "  make linear_square_error_example  # Build only example"
	@echo ""
	@echo "Currently available algorithms:"
	@for algo in $(ALGO_NAMES); do echo "  - $$algo"; done

clean:
	rm -rf $(BUILDDIR)
