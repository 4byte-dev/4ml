#include "linear_square_error.h"
#include "../../../tensor/cpu_engine.h"
#include <chrono>
#include <cstdio>
#include <random>

using namespace ml;
using Clock = std::chrono::high_resolution_clock;

int main() {
    CpuTensorEngine<float> engine;

    size_t n_samples = 100;
    size_t n_features = 1;
    float true_w = 3.0f;
    float true_b = 2.0f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    Tensor<float> X(engine, n_samples * n_features);
    Tensor<float> y(engine, n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        float x_val = static_cast<float>(i) / n_samples;
        X[i] = x_val;
        y[i] = true_w * x_val + true_b + noise(rng);
    }

    LinearSquareError<float> model(engine);

    auto start = Clock::now();
    model.fit(X, y, n_samples, n_features);
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("=== Linear Square Error (Least Squares) ===\n\n");
    printf("True:  y = %.1f * x + %.1f\n", true_w, true_b);
    printf("Learned: y = %.4f * x + %.4f\n", model.weights()[0], model.bias());
    printf("Training time: %.2f ms\n\n", ms);

    Tensor<float> preds = model.predict(X, n_samples, n_features);

    float mse = 0.0f;
    float rss = 0.0f;
    for (size_t i = 0; i < n_samples; ++i) {
        float diff = preds[i] - y[i];
        mse += diff * diff;
        rss += diff * diff;
    }
    mse /= n_samples;
    printf("MSE on training data: %.6f\n", mse);
    printf("RSS (Residual Sum of Squares): %.6f\n", rss);

    return 0;
}
