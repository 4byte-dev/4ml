#include "cpu_engine.h"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace ml {

template <typename T>
T* CpuTensorEngine<T>::malloc(size_t n) {
    T* ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
    if (!ptr && n > 0) throw std::bad_alloc();
    return ptr;
}

template <typename T>
void CpuTensorEngine<T>::free(T* ptr) { std::free(ptr); }

template <typename T>
void CpuTensorEngine<T>::memcpy(T* dst, const T* src, size_t n) {
    std::memcpy(dst, src, n * sizeof(T));
}

template <typename T>
void CpuTensorEngine<T>::fill(T* dst, T value, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = value;
}

template <typename T>
void CpuTensorEngine<T>::add(const T* a, const T* b, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] + b[i];
}

template <typename T>
void CpuTensorEngine<T>::sub(const T* a, const T* b, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] - b[i];
}

template <typename T>
void CpuTensorEngine<T>::elementwise_mul(const T* a, const T* b, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] * b[i];
}

template <typename T>
void CpuTensorEngine<T>::elementwise_div(const T* a, const T* b, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] / b[i];
}

template <typename T>
void CpuTensorEngine<T>::scale(const T* a, T alpha, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = alpha * a[i];
}

template <typename T>
void CpuTensorEngine<T>::add_scalar(const T* a, T alpha, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] + alpha;
}

template <typename T>
T CpuTensorEngine<T>::dot(const T* x, const T* y, size_t n) {
    T result = T(0);
    for (size_t i = 0; i < n; ++i) result += x[i] * y[i];
    return result;
}

template <typename T>
void CpuTensorEngine<T>::axpy(T alpha, const T* x, T* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

template <typename T>
T CpuTensorEngine<T>::nrm2(const T* x, size_t n) {
    T ss = T(0);
    for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    return static_cast<T>(std::sqrt(static_cast<double>(ss)));
}

template <typename T>
T CpuTensorEngine<T>::asum(const T* x, size_t n) {
    T s = T(0);
    for (size_t i = 0; i < n; ++i) {
        T v = x[i];
        s += (v < T(0)) ? -v : v;
    }
    return s;
}

template <typename T>
size_t CpuTensorEngine<T>::iamax(const T* x, size_t n) {
    if (n == 0) throw std::invalid_argument("iamax on empty");
    size_t idx = 0;
    T best = (x[0] < T(0)) ? -x[0] : x[0];
    for (size_t i = 1; i < n; ++i) {
        T v = (x[i] < T(0)) ? -x[i] : x[i];
        if (v > best) { best = v; idx = i; }
    }
    return idx;
}

template <typename T>
void CpuTensorEngine<T>::copy(const T* src, T* dst, size_t n) {
    std::memcpy(dst, src, n * sizeof(T));
}

template <typename T>
void CpuTensorEngine<T>::swap(T* x, T* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}

template <typename T>
void CpuTensorEngine<T>::rot(T* x, T* y, size_t n, T c, T s) {
    for (size_t i = 0; i < n; ++i) {
        T xi = c * x[i] + s * y[i];
        T yi = -s * x[i] + c * y[i];
        x[i] = xi; y[i] = yi;
    }
}

template <typename T>
void CpuTensorEngine<T>::gemv(Trans trans, size_t m, size_t n,
                               T alpha, const T* A, const T* x,
                               T beta, T* y) {
    if (trans == Trans::No) {
        // y = alpha * A * x + beta * y,  A is m x n
        for (size_t i = 0; i < m; ++i) {
            T dot = T(0);
            for (size_t j = 0; j < n; ++j) dot += A[i * n + j] * x[j];
            y[i] = alpha * dot + beta * y[i];
        }
    } else {
        // y = alpha * A^T * x + beta * y,  A is m x n, y is n
        for (size_t j = 0; j < n; ++j) {
            T dot = T(0);
            for (size_t i = 0; i < m; ++i) dot += A[i * n + j] * x[i];
            y[j] = alpha * dot + beta * y[j];
        }
    }
}

template <typename T>
void CpuTensorEngine<T>::ger(size_t m, size_t n,
                              T alpha, const T* x, const T* y, T* A) {
    // A += alpha * x * y^T
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            A[i * n + j] += alpha * x[i] * y[j];
}

template <typename T>
void CpuTensorEngine<T>::symv(Uplo uplo, size_t m,
                               T alpha, const T* A, const T* x,
                               T beta, T* y) {
    // y = alpha * A * x + beta * y, A symmetric m x m stored as full m x m
    for (size_t i = 0; i < m; ++i) {
        T sum = T(0);
        if (uplo == Uplo::Lower) {
            for (size_t j = 0; j <= i; ++j) sum += A[i * m + j] * x[j];
            for (size_t j = i + 1; j < m; ++j) sum += A[j * m + i] * x[j];
        } else {
            for (size_t j = 0; j < i; ++j) sum += A[j * m + i] * x[j];
            for (size_t j = i; j < m; ++j) sum += A[i * m + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

template <typename T>
void CpuTensorEngine<T>::trmv(Uplo uplo, Trans trans, size_t m,
                               const T* A, T* x) {
    // x = op(A) * x, A triangular m x m
    T* tmp = this->malloc(m);
    for (size_t i = 0; i < m; ++i) tmp[i] = T(0);

    if (uplo == Uplo::Upper) {
        if (trans == Trans::No) {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = i; j < m; ++j)
                    tmp[i] += A[i * m + j] * x[j];
        } else {
            for (size_t j = 0; j < m; ++j)
                for (size_t i = 0; i <= j; ++i)
                    tmp[j] += A[i * m + j] * x[i];
        }
    } else {
        if (trans == Trans::No) {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j <= i; ++j)
                    tmp[i] += A[i * m + j] * x[j];
        } else {
            for (size_t j = 0; j < m; ++j)
                for (size_t i = j; i < m; ++i)
                    tmp[j] += A[i * m + j] * x[i];
        }
    }

    this->memcpy(x, tmp, m);
    this->free(tmp);
}
template <typename T>
void CpuTensorEngine<T>::gemm(Trans transA, Trans transB,
                               size_t m, size_t n, size_t k,
                               T alpha, const T* A, const T* B,
                               T beta, T* C) {
    // C = alpha * op(A) * op(B) + beta * C
    // All stored row-major
    bool tA = (transA == Trans::Yes);
    bool tB = (transB == Trans::Yes);

    // Scale C by beta
    for (size_t i = 0; i < m * n; ++i) C[i] *= beta;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t p = 0; p < k; ++p) {
                T a = tA ? A[p * m + i] : A[i * k + p];
                T b = tB ? B[j * k + p] : B[p * n + j];
                sum += a * b;
            }
            C[i * n + j] += alpha * sum;
        }
    }
}

template <typename T>
void CpuTensorEngine<T>::syrk(Uplo uplo, size_t n, size_t k,
                               T alpha, const T* A, T beta, T* C) {
    // C = alpha * A * A^T + beta * C, C symmetric n x n, A is n x k
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            bool process = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
            if (process) {
                T sum = T(0);
                for (size_t p = 0; p < k; ++p) sum += A[i * k + p] * A[j * k + p];
                C[i * n + j] = alpha * sum + beta * C[i * n + j];
            }
        }
    }
}

template <typename T>
void CpuTensorEngine<T>::symm(Uplo uplo, size_t m, size_t n,
                               T alpha, const T* A, const T* B,
                               T beta, T* C) {
    // C = alpha * A * B + beta * C, A symmetric m x m, B m x n
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t p = 0; p < m; ++p) {
                T a;
                if (uplo == Uplo::Lower)
                    a = (i >= p) ? A[i * m + p] : A[p * m + i];
                else
                    a = (i <= p) ? A[i * m + p] : A[p * m + i];
                sum += a * B[p * n + j];
            }
            C[i * n + j] = alpha * sum + beta * C[i * n + j];
        }
    }
}

template <typename T>
void CpuTensorEngine<T>::trmm(Uplo uplo, Trans trans, size_t m, size_t n,
                               T alpha, const T* A, T* B) {
    // B = alpha * op(A) * B, A triangular m x m, B m x n
    T* tmp = this->malloc(m * n);
    for (size_t i = 0; i < m * n; ++i) tmp[i] = T(0);

    bool tA = (trans == Trans::Yes);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t p = 0; p < m; ++p) {
                bool use = (uplo == Uplo::Lower) ? (i >= p) : (i <= p);
                if (!use) continue;
                T a = tA ? A[p * m + i] : A[i * m + p];
                sum += a * B[p * n + j];
            }
            tmp[i * n + j] = alpha * sum;
        }
    }
    this->memcpy(B, tmp, m * n);
    this->free(tmp);
}

template <typename T>
void CpuTensorEngine<T>::trsm(Uplo uplo, Trans trans, size_t m, size_t n,
                               T alpha, const T* A, T* B) {
    // Solve op(A) * X = alpha * B, store X in B
    // A triangular m x m, B m x n
    // Scale B by alpha
    for (size_t i = 0; i < m * n; ++i) B[i] *= alpha;

    bool tA = (trans == Trans::Yes);

    if (uplo == Uplo::Lower) {
        if (!tA) {
            // Forward substitution: A * X = B, A lower triangular
            for (size_t col = 0; col < n; ++col) {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < i; ++j)
                        B[i * n + col] -= A[i * m + j] * B[j * n + col];
                    B[i * n + col] /= A[i * m + i];
                }
            }
        } else {
            // A^T * X = B, A^T is upper triangular
            for (size_t col = 0; col < n; ++col) {
                for (size_t i = m; i-- > 0;) {
                    for (size_t j = i + 1; j < m; ++j)
                        B[i * n + col] -= A[j * m + i] * B[j * n + col];
                    B[i * n + col] /= A[i * m + i];
                }
            }
        }
    } else {
        if (!tA) {
            // Back substitution: A * X = B, A upper triangular
            for (size_t col = 0; col < n; ++col) {
                for (size_t i = m; i-- > 0;) {
                    for (size_t j = i + 1; j < m; ++j)
                        B[i * n + col] -= A[i * m + j] * B[j * n + col];
                    B[i * n + col] /= A[i * m + i];
                }
            }
        } else {
            // A^T * X = B, A^T is lower triangular
            for (size_t col = 0; col < n; ++col) {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < i; ++j)
                        B[i * n + col] -= A[j * m + i] * B[j * n + col];
                    B[i * n + col] /= A[i * m + i];
                }
            }
        }
    }
}

template <typename T>
T CpuTensorEngine<T>::sum(const T* a, size_t n) {
    T r = T(0); for (size_t i = 0; i < n; ++i) r += a[i]; return r;
}

template <typename T>
T CpuTensorEngine<T>::max(const T* a, size_t n) {
    if (n == 0) throw std::invalid_argument("max on empty");
    T r = a[0]; for (size_t i = 1; i < n; ++i) if (a[i] > r) r = a[i]; return r;
}

template <typename T>
T CpuTensorEngine<T>::min(const T* a, size_t n) {
    if (n == 0) throw std::invalid_argument("min on empty");
    T r = a[0]; for (size_t i = 1; i < n; ++i) if (a[i] < r) r = a[i]; return r;
}

template <typename T>
size_t CpuTensorEngine<T>::argmax(const T* a, size_t n) {
    if (n == 0) throw std::invalid_argument("argmax on empty");
    size_t idx = 0;
    for (size_t i = 1; i < n; ++i) if (a[i] > a[idx]) idx = i;
    return idx;
}

template <typename T>
size_t CpuTensorEngine<T>::argmin(const T* a, size_t n) {
    if (n == 0) throw std::invalid_argument("argmin on empty");
    size_t idx = 0;
    for (size_t i = 1; i < n; ++i) if (a[i] < a[idx]) idx = i;
    return idx;
}

template <typename T>
void CpuTensorEngine<T>::transpose(const T* src, T* dst, size_t m, size_t n) {
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            dst[j * m + i] = src[i * n + j];
}

template <typename T>
void CpuTensorEngine<T>::exp(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(std::exp(static_cast<double>(src[i])));
}
template <typename T>
void CpuTensorEngine<T>::log(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(std::log(static_cast<double>(src[i])));
}
template <typename T>
void CpuTensorEngine<T>::abs(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = (src[i] < T(0)) ? -src[i] : src[i];
}
template <typename T>
void CpuTensorEngine<T>::sqrt(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(std::sqrt(static_cast<double>(src[i])));
}
template <typename T>
void CpuTensorEngine<T>::neg(const T* src, T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = -src[i];
}

template <typename T>
void CpuTensorEngine<T>::sum_axis(const T* src, T* dst, size_t rows, size_t cols, int axis) {
    if (axis == 0) {
        for (size_t j = 0; j < cols; ++j) { dst[j] = T(0); for (size_t i = 0; i < rows; ++i) dst[j] += src[i * cols + j]; }
    } else {
        for (size_t i = 0; i < rows; ++i) { dst[i] = T(0); for (size_t j = 0; j < cols; ++j) dst[i] += src[i * cols + j]; }
    }
}

template <typename T>
void CpuTensorEngine<T>::max_axis(const T* src, T* dst, size_t rows, size_t cols, int axis) {
    if (axis == 0) {
        for (size_t j = 0; j < cols; ++j) { dst[j] = src[j]; for (size_t i = 1; i < rows; ++i) if (src[i * cols + j] > dst[j]) dst[j] = src[i * cols + j]; }
    } else {
        for (size_t i = 0; i < rows; ++i) { dst[i] = src[i * cols]; for (size_t j = 1; j < cols; ++j) if (src[i * cols + j] > dst[i]) dst[i] = src[i * cols + j]; }
    }
}

template <typename T>
void CpuTensorEngine<T>::inverse(const T* A, T* dst, size_t n) {
    for (size_t i = 0; i < n * n; ++i) dst[i] = A[i];
    T* aug = this->malloc(n * n);
    eye(aug, n);

    for (size_t col = 0; col < n; ++col) {
        size_t max_row = col;
        for (size_t row = col + 1; row < n; ++row) {
            T va = (dst[row * n + col] < T(0)) ? -dst[row * n + col] : dst[row * n + col];
            T vb = (dst[max_row * n + col] < T(0)) ? -dst[max_row * n + col] : dst[max_row * n + col];
            if (va > vb) max_row = row;
        }
        if (max_row != col) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(dst[col * n + j], dst[max_row * n + j]);
                std::swap(aug[col * n + j], aug[max_row * n + j]);
            }
        }
        T pivot = dst[col * n + col];
        if (pivot == T(0)) throw std::runtime_error("Matrix is singular");
        T inv_p = T(1) / pivot;
        for (size_t j = 0; j < n; ++j) { dst[col * n + j] *= inv_p; aug[col * n + j] *= inv_p; }
        for (size_t row = 0; row < n; ++row) {
            if (row == col) continue;
            T f = dst[row * n + col];
            for (size_t j = 0; j < n; ++j) { dst[row * n + j] -= f * dst[col * n + j]; aug[row * n + j] -= f * aug[col * n + j]; }
        }
    }
    this->memcpy(dst, aug, n * n);
    this->free(aug);
}

template <typename T>
T CpuTensorEngine<T>::determinant(const T* A, size_t n) {
    T* tmp = this->malloc(n * n);
    this->memcpy(tmp, A, n * n);
    T det = T(1);
    for (size_t col = 0; col < n; ++col) {
        size_t max_row = col;
        for (size_t row = col + 1; row < n; ++row) {
            T va = (tmp[row * n + col] < T(0)) ? -tmp[row * n + col] : tmp[row * n + col];
            T vb = (tmp[max_row * n + col] < T(0)) ? -tmp[max_row * n + col] : tmp[max_row * n + col];
            if (va > vb) max_row = row;
        }
        if (max_row != col) { for (size_t j = 0; j < n; ++j) std::swap(tmp[col * n + j], tmp[max_row * n + j]); det = -det; }
        T pivot = tmp[col * n + col];
        if (pivot == T(0)) { this->free(tmp); return T(0); }
        det *= pivot;
        for (size_t row = col + 1; row < n; ++row) {
            T f = tmp[row * n + col] / pivot;
            for (size_t j = col + 1; j < n; ++j) tmp[row * n + j] -= f * tmp[col * n + j];
        }
    }
    this->free(tmp);
    return det;
}

template <typename T>
void CpuTensorEngine<T>::eye(T* dst, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            dst[i * n + j] = (i == j) ? T(1) : T(0);
}

template class CpuTensorEngine<float>;
template class CpuTensorEngine<double>;
template class CpuTensorEngine<int8_t>;
template class CpuTensorEngine<uint8_t>;
template class CpuTensorEngine<int16_t>;
template class CpuTensorEngine<uint16_t>;
template class CpuTensorEngine<int32_t>;
template class CpuTensorEngine<uint32_t>;
template class CpuTensorEngine<int64_t>;
template class CpuTensorEngine<uint64_t>;

} // namespace ml
