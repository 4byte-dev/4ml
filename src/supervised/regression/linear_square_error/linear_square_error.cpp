#include "linear_square_error.h"

namespace ml {

template <typename T>
void LinearSquareError<T>::fit(Tensor<T>& X, Tensor<T>& y, size_t n_samples, size_t n_features) {
    Tensor<T> X_aug(engine_, n_samples, n_features + 1);
    for (size_t i = 0; i < n_samples; ++i) {
        X_aug(i, 0) = T(1);
        for (size_t j = 0; j < n_features; ++j) {
            X_aug(i, j + 1) = X[i * n_features + j];
        }
    }
    
    Tensor<T> y_col = y.reshape(n_samples, 1);
    
    Tensor<T> Xt = X_aug.transpose();
    Tensor<T> XtX = Xt.matmul(X_aug);
    Tensor<T> XtX_inv = XtX.inv();
    Tensor<T> Xty = Xt.matmul(y_col);
    Tensor<T> result = XtX_inv.matmul(Xty);
    
    bias_ = result(0, 0);
    weights_ = Tensor<T>(engine_, n_features);
    for (size_t j = 0; j < n_features; ++j) {
        weights_[j] = result(j + 1, 0);
    }
}

template <typename T>
Tensor<T> LinearSquareError<T>::predict(Tensor<T>& X, size_t n_samples, size_t n_features) {
    Tensor<T> y_pred(engine_, n_samples);
    engine_.gemv(Trans::No, n_samples, n_features, T(1), X.data(), weights_.data(), T(0), y_pred.data());
    engine_.add_scalar(y_pred.data(), bias_, y_pred.data(), n_samples);
    return y_pred;
}

template class LinearSquareError<float>;
template class LinearSquareError<double>;

} // namespace ml
