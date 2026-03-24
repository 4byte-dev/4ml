#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace ml {

enum class Trans { No, Yes };

enum class Uplo { Upper, Lower };

template <typename T>
class TensorEngine {
public:
    virtual ~TensorEngine() = default;

    virtual T* malloc(size_t n) = 0;
    virtual void free(T* ptr) = 0;
    virtual void memcpy(T* dst, const T* src, size_t n) = 0;

    virtual void fill(T* dst, T value, size_t n) = 0;

    virtual void add(const T* a, const T* b, T* dst, size_t n) = 0;
    virtual void sub(const T* a, const T* b, T* dst, size_t n) = 0;
    virtual void elementwise_mul(const T* a, const T* b, T* dst, size_t n) = 0;
    virtual void elementwise_div(const T* a, const T* b, T* dst, size_t n) = 0;

    virtual void scale(const T* a, T alpha, T* dst, size_t n) = 0;
    virtual void add_scalar(const T* a, T alpha, T* dst, size_t n) = 0;

    virtual T   dot(const T* x, const T* y, size_t n) = 0;
    virtual void axpy(T alpha, const T* x, T* y, size_t n) = 0;
    virtual T   nrm2(const T* x, size_t n) = 0;            // sqrt(sum(x[i]^2))
    virtual T   asum(const T* x, size_t n) = 0;             // sum(abs(x[i]))
    virtual size_t iamax(const T* x, size_t n) = 0;         // index of max(abs(x[i]))
    virtual void copy(const T* src, T* dst, size_t n) = 0;  // dst = src
    virtual void swap(T* x, T* y, size_t n) = 0;            // swap x <-> y
    virtual void rot(T* x, T* y, size_t n, T c, T s) = 0;  // Givens rotation

    // gemv: y = alpha * op(A) * x + beta * y
    // A is m x n (row-major), op(A) is trans(A) ? A^T : A
    virtual void gemv(Trans trans, size_t m, size_t n,
                      T alpha, const T* A, const T* x,
                      T beta, T* y) = 0;

    // ger: A = alpha * x * y^T + A  (rank-1 update)
    // A is m x n (row-major)
    virtual void ger(size_t m, size_t n,
                     T alpha, const T* x, const T* y, T* A) = 0;

    // symv: y = alpha * A * x + beta * y  (A symmetric m x m)
    virtual void symv(Uplo uplo, size_t m,
                      T alpha, const T* A, const T* x,
                      T beta, T* y) = 0;

    // trmv: x = op(A) * x  (A triangular m x m, in-place)
    virtual void trmv(Uplo uplo, Trans trans, size_t m,
                      const T* A, T* x) = 0;

    // gemm: C = alpha * op(A) * op(B) + beta * C
    // A is m x k, B is k x n, C is m x n (row-major)
    virtual void gemm(Trans transA, Trans transB,
                      size_t m, size_t n, size_t k,
                      T alpha, const T* A, const T* B,
                      T beta, T* C) = 0;

    // syrk: C = alpha * A * A^T + beta * C  (symmetric rank-k update)
    // C is n x n symmetric, A is n x k (row-major)
    virtual void syrk(Uplo uplo, size_t n, size_t k,
                      T alpha, const T* A, T beta, T* C) = 0;

    // symm: C = alpha * A * B + beta * C  (A symmetric m x m, B m x n)
    virtual void symm(Uplo uplo, size_t m, size_t n,
                      T alpha, const T* A, const T* B,
                      T beta, T* C) = 0;

    // trmm: B = alpha * op(A) * B  (A triangular m x m, B m x n)
    virtual void trmm(Uplo uplo, Trans trans, size_t m, size_t n,
                      T alpha, const T* A, T* B) = 0;

    // trsm: Solve op(A) * X = alpha * B, storing result in B
    // A triangular m x m, B m x n
    virtual void trsm(Uplo uplo, Trans trans, size_t m, size_t n,
                      T alpha, const T* A, T* B) = 0;

    virtual T sum(const T* a, size_t n) = 0;
    virtual T max(const T* a, size_t n) = 0;
    virtual T min(const T* a, size_t n) = 0;
    virtual size_t argmax(const T* a, size_t n) = 0;
    virtual size_t argmin(const T* a, size_t n) = 0;

    virtual void transpose(const T* src, T* dst, size_t m, size_t n) = 0;

    virtual void exp(const T* src, T* dst, size_t n) = 0;
    virtual void log(const T* src, T* dst, size_t n) = 0;
    virtual void abs(const T* src, T* dst, size_t n) = 0;
    virtual void sqrt(const T* src, T* dst, size_t n) = 0;
    virtual void neg(const T* src, T* dst, size_t n) = 0;

    virtual void sum_axis(const T* src, T* dst, size_t rows, size_t cols, int axis) = 0;
    virtual void max_axis(const T* src, T* dst, size_t rows, size_t cols, int axis) = 0;

    virtual void inverse(const T* A, T* dst, size_t n) = 0;
    virtual T determinant(const T* A, size_t n) = 0;
    virtual void eye(T* dst, size_t n) = 0;
};

} // namespace ml
