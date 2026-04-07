#pragma once

#include "../../../tensor/tensor.h"

namespace ml {

template <typename T>
class LinearSquareError {
public:
    LinearSquareError(TensorEngine<T>& engine) : engine_(engine), bias_(T(0)) {}

    void fit(Tensor<T>& X, Tensor<T>& y, size_t n_samples, size_t n_features);
    Tensor<T> predict(Tensor<T>& X, size_t n_samples, size_t n_features);

    const Tensor<T>& weights() const { return weights_; }
    T bias() const { return bias_; }

private:
    TensorEngine<T>& engine_;
    Tensor<T> weights_{engine_, 0};
    T bias_;
};

} // namespace ml
