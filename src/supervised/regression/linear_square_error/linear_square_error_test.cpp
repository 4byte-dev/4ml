#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "tensor/tensor.h"
#include "tensor/cpu_engine.h"
#include "linear_square_error.h"

using namespace ml;

static constexpr float kTol = 1e-4f;

class LinearSquareErrorTest : public ::testing::Test {
protected:
    CpuTensorEngine<float> eng;
};

TEST_F(LinearSquareErrorTest, SimpleLinearFit) {
    size_t n_samples = 10;
    size_t n_features = 1;
    float true_w = 3.0f;
    float true_b = 2.0f;

    Tensor<float> X(eng, n_samples * n_features);
    Tensor<float> y(eng, n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        float x_val = static_cast<float>(i) / n_samples;
        X[i] = x_val;
        y[i] = true_w * x_val + true_b;
    }

    LinearSquareError<float> model(eng);
    model.fit(X, y, n_samples, n_features);

    EXPECT_NEAR(model.weights()[0], true_w, kTol);
    EXPECT_NEAR(model.bias(), true_b, kTol);
}

TEST_F(LinearSquareErrorTest, MultiFeature) {
    size_t n_samples = 50;
    size_t n_features = 2;
    float true_w1 = 2.5f;
    float true_w2 = -1.5f;
    float true_b = 1.0f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Tensor<float> X(eng, n_samples * n_features);
    Tensor<float> y(eng, n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        float x1 = dist(rng);
        float x2 = dist(rng);
        X[i * n_features] = x1;
        X[i * n_features + 1] = x2;
        y[i] = true_w1 * x1 + true_w2 * x2 + true_b;
    }

    LinearSquareError<float> model(eng);
    model.fit(X, y, n_samples, n_features);

    EXPECT_NEAR(model.weights()[0], true_w1, kTol);
    EXPECT_NEAR(model.weights()[1], true_w2, kTol);
}

TEST_F(LinearSquareErrorTest, RSS_Calculation) {
    size_t n_samples = 10;
    size_t n_features = 1;

    Tensor<float> X(eng, n_samples * n_features);
    Tensor<float> y(eng, n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        X[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i * 2 + 5);
    }

    LinearSquareError<float> model(eng);
    model.fit(X, y, n_samples, n_features);

    Tensor<float> preds = model.predict(X, n_samples, n_features);

    float rss = 0.0f;
    for (size_t i = 0; i < n_samples; ++i) {
        float residual = y[i] - preds[i];
        rss += residual * residual;
    }

    EXPECT_LT(rss, kTol);
}

TEST_F(LinearSquareErrorTest, MSE_Training) {
    size_t n_samples = 100;
    size_t n_features = 1;
    float true_w = 2.0f;
    float true_b = 1.0f;

    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    Tensor<float> X(eng, n_samples * n_features);
    Tensor<float> y(eng, n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        float x_val = static_cast<float>(i) / n_samples;
        X[i] = x_val;
        y[i] = true_w * x_val + true_b + noise(rng);
    }

    LinearSquareError<float> model(eng);
    model.fit(X, y, n_samples, n_features);

    Tensor<float> preds = model.predict(X, n_samples, n_features);

    float mse = 0.0f;
    for (size_t i = 0; i < n_samples; ++i) {
        float diff = preds[i] - y[i];
        mse += diff * diff;
    }
    mse /= n_samples;

    EXPECT_LT(mse, 1.0f);
}
