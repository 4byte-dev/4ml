#pragma once

#include "engine.h"
#include <initializer_list>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace ml {

template <typename T>
class Tensor {
public:
    Tensor(TensorEngine<T>& engine, size_t size)
        : engine_(engine), data_(engine.malloc(size)),
          size_(size), rows_(1), cols_(size), owned_(true) {}

    Tensor(TensorEngine<T>& engine, size_t rows, size_t cols)
        : engine_(engine), data_(engine.malloc(rows * cols)),
          size_(rows * cols), rows_(rows), cols_(cols), owned_(true) {}

    Tensor(TensorEngine<T>& engine, std::initializer_list<T> values)
        : engine_(engine), size_(values.size()),
          rows_(1), cols_(values.size()), owned_(true) {
        data_ = engine_.malloc(size_);
        size_t i = 0;
        for (auto& v : values) data_[i++] = v;
    }

    Tensor(TensorEngine<T>& engine, const std::vector<T>& values)
        : engine_(engine), size_(values.size()),
          rows_(1), cols_(values.size()), owned_(true) {
        data_ = engine_.malloc(size_);
        engine_.memcpy(data_, values.data(), size_);
    }

    Tensor(TensorEngine<T>& engine, T* data, size_t size, bool owned = false)
        : engine_(engine), data_(data), size_(size), rows_(1), cols_(size), owned_(owned) {}

    Tensor(TensorEngine<T>& engine, T* data, size_t rows, size_t cols, bool owned = false)
        : engine_(engine), data_(data), size_(rows * cols), rows_(rows), cols_(cols), owned_(owned) {}

    Tensor(const Tensor& other)
        : engine_(other.engine_), size_(other.size_),
          rows_(other.rows_), cols_(other.cols_), owned_(true) {
        data_ = engine_.malloc(size_);
        engine_.memcpy(data_, other.data_, size_);
    }

    Tensor(Tensor&& other) noexcept
        : engine_(other.engine_), data_(other.data_), size_(other.size_),
          rows_(other.rows_), cols_(other.cols_), owned_(other.owned_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.owned_ = false;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (owned_ && data_) engine_.free(data_);
            size_ = other.size_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            owned_ = true;
            data_ = engine_.malloc(size_);
            engine_.memcpy(data_, other.data_, size_);
        }
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owned_ && data_) engine_.free(data_);
            data_ = other.data_;
            size_ = other.size_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            owned_ = other.owned_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.owned_ = false;
        }
        return *this;
    }

    ~Tensor() {
        if (owned_ && data_) engine_.free(data_);
    }

    size_t size() const { return size_; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T* data() { return data_; }
    const T* data() const { return data_; }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T& operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }
    const T& operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }

    TensorEngine<T>& engine() { return engine_; }

    bool is_matrix() const { return rows_ > 1; }
    bool is_vector() const { return rows_ == 1; }

    Tensor reshape(size_t new_rows, size_t new_cols) const {
        if (new_rows * new_cols != size_)
            throw std::invalid_argument("reshape: total size mismatch");
        return Tensor(engine_, data_, new_rows, new_cols, false);
    }

    Tensor transpose() const {
        Tensor result(engine_, cols_, rows_);
        engine_.transpose(data_, result.data_, rows_, cols_);
        return result;
    }

    Tensor row(size_t i) const {
        Tensor result(engine_, cols_);
        engine_.memcpy(result.data(), data_ + i * cols_, cols_);
        return result;
    }

    Tensor col(size_t j) const {
        Tensor result(engine_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            result[i] = data_[i * cols_ + j];
        return result;
    }

    Tensor block(size_t r, size_t c, size_t block_rows, size_t block_cols) const {
        if (r + block_rows > rows_ || c + block_cols > cols_)
            throw std::invalid_argument("block: out of bounds");
        Tensor result(engine_, block_rows, block_cols);
        for (size_t i = 0; i < block_rows; ++i)
            engine_.memcpy(result.data() + i * block_cols,
                          data_ + (r + i) * cols_ + c, block_cols);
        return result;
    }

    static Tensor hstack(const Tensor& a, const Tensor& b) {
        if (a.rows_ != b.rows_)
            throw std::invalid_argument("hstack: row count mismatch");
        Tensor result(a.engine_, a.rows_, a.cols_ + b.cols_);
        for (size_t i = 0; i < a.rows_; ++i) {
            a.engine_.memcpy(result.data_ + i * result.cols_, a.data_ + i * a.cols_, a.cols_);
            a.engine_.memcpy(result.data_ + i * result.cols_ + a.cols_, b.data_ + i * b.cols_, b.cols_);
        }
        return result;
    }

    static Tensor vstack(const Tensor& a, const Tensor& b) {
        if (a.cols_ != b.cols_)
            throw std::invalid_argument("vstack: col count mismatch");
        Tensor result(a.engine_, a.rows_ + b.rows_, a.cols_);
        a.engine_.memcpy(result.data_, a.data_, a.size_);
        a.engine_.memcpy(result.data_ + a.size_, b.data_, b.size_);
        return result;
    }

    Tensor operator+(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in +");
        Tensor result(engine_, rows_, cols_);
        engine_.add(data_, other.data_, result.data_, size_);
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in -");
        Tensor result(engine_, rows_, cols_);
        engine_.sub(data_, other.data_, result.data_, size_);
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in *");
        Tensor result(engine_, rows_, cols_);
        engine_.elementwise_mul(data_, other.data_, result.data_, size_);
        return result;
    }

    Tensor operator/(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in /");
        Tensor result(engine_, rows_, cols_);
        engine_.elementwise_div(data_, other.data_, result.data_, size_);
        return result;
    }

    Tensor operator+(T scalar) const {
        Tensor result(engine_, rows_, cols_);
        engine_.add_scalar(data_, scalar, result.data_, size_);
        return result;
    }

    Tensor operator-(T scalar) const { return *this + (-scalar); }

    Tensor operator*(T scalar) const {
        Tensor result(engine_, rows_, cols_);
        engine_.scale(data_, scalar, result.data_, size_);
        return result;
    }

    Tensor operator/(T scalar) const {
        return *this * (T(1) / scalar);
    }

    friend Tensor operator+(T scalar, const Tensor& t) {
        return t + scalar;
    }

    friend Tensor operator-(T scalar, const Tensor& t) {
        // scalar - t: negate each element, then add scalar
        Tensor result(t.engine_, t.rows_, t.cols_);
        t.engine_.neg(t.data_, result.data_, t.size_);
        t.engine_.add_scalar(result.data(), scalar, result.data(), t.size_);
        return result;
    }

    friend Tensor operator*(T scalar, const Tensor& t) {
        return t * scalar;
    }

    friend Tensor operator/(T scalar, const Tensor& t) {
        Tensor result(t.engine_, t.rows_, t.cols_);
        for (size_t i = 0; i < t.size_; ++i)
            result.data_[i] = scalar / t.data_[i];
        return result;
    }

    Tensor& operator+=(const Tensor& other) {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in +=");
        engine_.axpy(T(1), other.data_, data_, size_);
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in -=");
        engine_.axpy(T(-1), other.data_, data_, size_);
        return *this;
    }

    Tensor& operator*=(const Tensor& other) {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in *=");
        engine_.elementwise_mul(data_, other.data_, data_, size_);
        return *this;
    }

    Tensor& operator/=(const Tensor& other) {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in /=");
        engine_.elementwise_div(data_, other.data_, data_, size_);
        return *this;
    }

    Tensor& operator+=(T scalar) {
        engine_.add_scalar(data_, scalar, data_, size_);
        return *this;
    }

    Tensor& operator-=(T scalar) {
        engine_.add_scalar(data_, -scalar, data_, size_);
        return *this;
    }

    Tensor& operator*=(T scalar) {
        engine_.scale(data_, scalar, data_, size_);
        return *this;
    }

    Tensor& operator/=(T scalar) {
        T inv = T(1) / scalar;
        engine_.scale(data_, inv, data_, size_);
        return *this;
    }

    Tensor operator-() const {
        Tensor result(engine_, rows_, cols_);
        engine_.neg(data_, result.data_, size_);
        return result;
    }

    Tensor operator+() const { return *this; }

    Tensor& operator++() {
        engine_.add_scalar(data_, T(1), data_, size_);
        return *this;
    }

    Tensor operator++(int) {
        Tensor tmp(*this);
        engine_.add_scalar(data_, T(1), data_, size_);
        return tmp;
    }

    Tensor& operator--() {
        engine_.add_scalar(data_, T(-1), data_, size_);
        return *this;
    }

    Tensor operator--(int) {
        Tensor tmp(*this);
        engine_.add_scalar(data_, T(-1), data_, size_);
        return tmp;
    }

    bool operator==(const Tensor& other) const {
        if (size_ != other.size_) return false;
        for (size_t i = 0; i < size_; ++i)
            if (data_[i] != other.data_[i]) return false;
        return true;
    }

    bool operator!=(const Tensor& other) const { return !(*this == other); }

    bool operator<(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in <");
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] < other.data_[i])) return false;
        return true;
    }

    bool operator>(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in >");
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] > other.data_[i])) return false;
        return true;
    }

    bool operator<=(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in <=");
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] <= other.data_[i])) return false;
        return true;
    }

    bool operator>=(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in >=");
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] >= other.data_[i])) return false;
        return true;
    }

    bool operator<(T scalar) const {
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] < scalar)) return false;
        return true;
    }

    bool operator>(T scalar) const {
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] > scalar)) return false;
        return true;
    }

    bool operator<=(T scalar) const {
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] <= scalar)) return false;
        return true;
    }

    bool operator>=(T scalar) const {
        for (size_t i = 0; i < size_; ++i)
            if (!(data_[i] >= scalar)) return false;
        return true;
    }

    Tensor matmul(const Tensor& B) const {
        if (cols_ != B.rows_)
            throw std::invalid_argument("matmul: dimension mismatch");
        Tensor result(engine_, rows_, B.cols_);
        engine_.gemm(Trans::No, Trans::No, rows_, B.cols_, cols_,
                     T(1), data_, B.data_, T(0), result.data_);
        return result;
    }

    Tensor gemm(const Tensor& B, T alpha = T(1), T beta = T(0)) const {
        if (cols_ != B.rows_)
            throw std::invalid_argument("gemm: dimension mismatch");
        Tensor result(engine_, rows_, B.cols_);
        result.fill(T(0));
        engine_.gemm(Trans::No, Trans::No, rows_, B.cols_, cols_,
                     alpha, data_, B.data_, beta, result.data_);
        return result;
    }

    Tensor matvec(const Tensor& x) const {
        if (cols_ != x.size_)
            throw std::invalid_argument("matvec: dimension mismatch");
        Tensor result(engine_, rows_);
        engine_.gemv(Trans::No, rows_, cols_, T(1), data_, x.data_, T(0), result.data_);
        return result;
    }

    Tensor gemv(const Tensor& x, T alpha = T(1), T beta = T(0)) const {
        if (cols_ != x.size_)
            throw std::invalid_argument("gemv: dimension mismatch");
        Tensor result(engine_, rows_);
        engine_.gemv(Trans::No, rows_, cols_, alpha, data_, x.data_, beta, result.data_);
        return result;
    }

    void ger(const Tensor& x, const Tensor& y, T alpha = T(1)) {
        if (x.rows_ != rows_ || y.rows_ != cols_)
            throw std::invalid_argument("ger: dimension mismatch");
        engine_.ger(rows_, cols_, alpha, x.data_, y.data_, data_);
    }

    Tensor syrk(T alpha = T(1), T beta = T(0)) const {
        Tensor result(engine_, rows_, rows_);
        result.fill(T(0));
        engine_.syrk(Uplo::Lower, rows_, cols_, alpha, data_, beta, result.data_);
        return result;
    }

    Tensor symm(const Tensor& B, T alpha = T(1), T beta = T(0)) const {
        if (rows_ != cols_)
            throw std::invalid_argument("symm: A must be square/symmetric");
        Tensor result(engine_, rows_, B.cols_);
        result.fill(T(0));
        engine_.symm(Uplo::Lower, rows_, B.cols_, alpha, data_, B.data_, beta, result.data_);
        return result;
    }

    Tensor trmm(const Tensor& B, T alpha = T(1)) const {
        if (rows_ != cols_)
            throw std::invalid_argument("trmm: A must be square/triangular");
        Tensor result(B);
        engine_.trmm(Uplo::Lower, Trans::No, rows_, B.cols_, alpha, data_, result.data_);
        return result;
    }

    Tensor inv() const {
        if (rows_ != cols_)
            throw std::invalid_argument("inv: matrix must be square");
        Tensor result(engine_, rows_, rows_);
        engine_.inverse(data_, result.data_, rows_);
        return result;
    }

    T det() const {
        if (rows_ != cols_)
            throw std::invalid_argument("det: matrix must be square");
        return engine_.determinant(data_, rows_);
    }

    T nrm2() const { return engine_.nrm2(data_, size_); }
    T asum() const { return engine_.asum(data_, size_); }
    size_t iamax() const { return engine_.iamax(data_, size_); }
    void copy_to(Tensor& dst) const {
        if (size_ != dst.size_) throw std::invalid_argument("copy_to: size mismatch");
        engine_.copy(data_, dst.data_, size_);
    }

    T sum() const { return engine_.sum(data_, size_); }
    T max() const { return engine_.max(data_, size_); }
    T min() const { return engine_.min(data_, size_); }
    size_t argmax() const { return engine_.argmax(data_, size_); }
    size_t argmin() const { return engine_.argmin(data_, size_); }

    T dot(const Tensor& other) const {
        if (size_ != other.size_) throw std::invalid_argument("Tensor size mismatch in dot");
        return engine_.dot(data_, other.data_, size_);
    }

    Tensor sum(int axis) const {
        if (axis == 0) {
            Tensor result(engine_, 1, cols_);
            engine_.sum_axis(data_, result.data_, rows_, cols_, 0);
            return result;
        } else {
            Tensor result(engine_, rows_, 1);
            engine_.sum_axis(data_, result.data_, rows_, cols_, 1);
            return result;
        }
    }

    Tensor max(int axis) const {
        if (axis == 0) {
            Tensor result(engine_, 1, cols_);
            engine_.max_axis(data_, result.data_, rows_, cols_, 0);
            return result;
        } else {
            Tensor result(engine_, rows_, 1);
            engine_.max_axis(data_, result.data_, rows_, cols_, 1);
            return result;
        }
    }

    Tensor exp() const {
        Tensor result(engine_, rows_, cols_);
        engine_.exp(data_, result.data_, size_);
        return result;
    }

    Tensor log() const {
        Tensor result(engine_, rows_, cols_);
        engine_.log(data_, result.data_, size_);
        return result;
    }

    Tensor abs() const {
        Tensor result(engine_, rows_, cols_);
        engine_.abs(data_, result.data_, size_);
        return result;
    }

    Tensor sqrt() const {
        Tensor result(engine_, rows_, cols_);
        engine_.sqrt(data_, result.data_, size_);
        return result;
    }

    Tensor neg() const {
        Tensor result(engine_, rows_, cols_);
        engine_.neg(data_, result.data_, size_);
        return result;
    }

    void fill(T value) { engine_.fill(data_, value, size_); }

    static Tensor eye(TensorEngine<T>& engine, size_t n) {
        Tensor result(engine, n, n);
        engine.eye(result.data_, n);
        return result;
    }

    static Tensor zeros(TensorEngine<T>& engine, size_t rows, size_t cols) {
        Tensor result(engine, rows, cols);
        result.fill(T(0));
        return result;
    }

    static Tensor ones(TensorEngine<T>& engine, size_t rows, size_t cols) {
        Tensor result(engine, rows, cols);
        result.fill(T(1));
        return result;
    }

    static Tensor constant(TensorEngine<T>& engine, size_t rows, size_t cols, T value) {
        Tensor result(engine, rows, cols);
        result.fill(value);
        return result;
    }

    static Tensor zeros(TensorEngine<T>& engine, size_t n) { return zeros(engine, 1, n); }
    static Tensor ones(TensorEngine<T>& engine, size_t n) { return ones(engine, 1, n); }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if (t.rows_ > 1) {
            os << "[";
            for (size_t i = 0; i < t.rows_; ++i) {
                if (i > 0) os << " ";
                os << "[";
                for (size_t j = 0; j < t.cols_; ++j) {
                    if (j > 0) os << ", ";
                    os << t.data_[i * t.cols_ + j];
                }
                os << "]";
                if (i < t.rows_ - 1) os << "\n";
            }
            os << "]";
        } else {
            os << "[";
            for (size_t i = 0; i < t.size_; ++i) {
                if (i > 0) os << ", ";
                os << t.data_[i];
            }
            os << "]";
        }
        return os;
    }

private:
    TensorEngine<T>& engine_;
    T* data_;
    size_t size_;
    size_t rows_;
    size_t cols_;
    bool owned_;
};

} // namespace ml
