#pragma once

#include "engine.h"

namespace ml {

template <typename T>
class CpuTensorEngine : public TensorEngine<T> {
public:
    T* malloc(size_t n) override;
    void free(T* ptr) override;
    void memcpy(T* dst, const T* src, size_t n) override;

    void fill(T* dst, T value, size_t n) override;

    void add(const T* a, const T* b, T* dst, size_t n) override;
    void sub(const T* a, const T* b, T* dst, size_t n) override;
    void elementwise_mul(const T* a, const T* b, T* dst, size_t n) override;
    void elementwise_div(const T* a, const T* b, T* dst, size_t n) override;

    void scale(const T* a, T alpha, T* dst, size_t n) override;
    void add_scalar(const T* a, T alpha, T* dst, size_t n) override;

    T dot(const T* x, const T* y, size_t n) override;
    void axpy(T alpha, const T* x, T* y, size_t n) override;
    T nrm2(const T* x, size_t n) override;
    T asum(const T* x, size_t n) override;
    size_t iamax(const T* x, size_t n) override;
    void copy(const T* src, T* dst, size_t n) override;
    void swap(T* x, T* y, size_t n) override;
    void rot(T* x, T* y, size_t n, T c, T s) override;

    void gemv(Trans trans, size_t m, size_t n,
              T alpha, const T* A, const T* x,
              T beta, T* y) override;
    void ger(size_t m, size_t n,
             T alpha, const T* x, const T* y, T* A) override;
    void symv(Uplo uplo, size_t m,
              T alpha, const T* A, const T* x,
              T beta, T* y) override;
    void trmv(Uplo uplo, Trans trans, size_t m,
              const T* A, T* x) override;

    void gemm(Trans transA, Trans transB,
              size_t m, size_t n, size_t k,
              T alpha, const T* A, const T* B,
              T beta, T* C) override;
    void syrk(Uplo uplo, size_t n, size_t k,
              T alpha, const T* A, T beta, T* C) override;
    void symm(Uplo uplo, size_t m, size_t n,
              T alpha, const T* A, const T* B,
              T beta, T* C) override;
    void trmm(Uplo uplo, Trans trans, size_t m, size_t n,
              T alpha, const T* A, T* B) override;
    void trsm(Uplo uplo, Trans trans, size_t m, size_t n,
              T alpha, const T* A, T* B) override;

    T sum(const T* a, size_t n) override;
    T max(const T* a, size_t n) override;
    T min(const T* a, size_t n) override;
    size_t argmax(const T* a, size_t n) override;
    size_t argmin(const T* a, size_t n) override;

    void transpose(const T* src, T* dst, size_t m, size_t n) override;

    void exp(const T* src, T* dst, size_t n) override;
    void log(const T* src, T* dst, size_t n) override;
    void abs(const T* src, T* dst, size_t n) override;
    void sqrt(const T* src, T* dst, size_t n) override;
    void neg(const T* src, T* dst, size_t n) override;

    void sum_axis(const T* src, T* dst, size_t rows, size_t cols, int axis) override;
    void max_axis(const T* src, T* dst, size_t rows, size_t cols, int axis) override;

    void inverse(const T* A, T* dst, size_t n) override;
    T determinant(const T* A, size_t n) override;
    void eye(T* dst, size_t n) override;
};

} // namespace ml
