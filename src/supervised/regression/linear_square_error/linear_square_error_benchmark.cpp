#include "linear_square_error.h"
#include "../../../tensor/cpu_engine.h"
#include <chrono>
#include <cstdio>
#include <random>

using namespace ml;
using Clock = std::chrono::high_resolution_clock;

void benchmark_lr(size_t n_samples, size_t n_features) {
    CpuTensorEngine<float> engine;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    Tensor<float> X(engine, n_samples * n_features);
    Tensor<float> y(engine, n_samples);

    for (size_t i = 0; i < n_samples * n_features; ++i)
        X[i] = dist(rng);
    for (size_t i = 0; i < n_samples; ++i)
        y[i] = dist(rng);

    LinearSquareError<float> model(engine);

    auto start = Clock::now();
    model.fit(X, y, n_samples, n_features);
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("  samples=%8zu  features=%4zu  time=%8.2f ms\n",
           n_samples, n_features, ms);
}

int main() {
    printf("=== Linear Square Error Benchmark ===\n\n");

    printf("Scaling samples (1 feature):\n");
    for (size_t n : {100, 500, 1000, 5000, 10000, 50000})
        benchmark_lr(n, 1);

    printf("\nScaling features (1000 samples):\n");
    for (size_t f : {1, 5, 10, 50, 100, 500})
        benchmark_lr(1000, f);

    return 0;
}
