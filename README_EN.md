# ML

A C++ machine learning library with pluggable tensor compute backends. The tensor engine is abstracted behind a clean interface, making it easy to swap between CPU, CUDA, OpenCL, or custom hardware backends without changing ML algorithm code.

## Features

- **Pluggable backend architecture** — swap `CpuTensorEngine` for any custom engine
- **Full BLAS Level 1–3** — `dot`, `axpy`, `gemv`, `ger`, `gemm`, `syrk`, `trmm`, `trsm`, ...
- **10 data types** — `float`, `double`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
- **Eigen-like tensor API** — operator overloading (`+`, `-`, `*`, `/`, `+=`, `-=`, comparisons, increment/decrement)
- **Shape-aware tensors** — 1D/2D with `reshape`, `transpose`, `row`, `col`, `block`, `hstack`, `vstack`
- **Element-wise math** — `exp`, `log`, `sqrt`, `abs`, `neg`
- **Axis reductions** — `sum(axis)`, `max(axis)` along rows or columns
- **Linear algebra** — `inverse`, `determinant`, `eye`
- **Benchmarks** — Per-operation timing for all types and BLAS levels

## Project Structure

```
ml/
├── CMakeLists.txt
├── Makefile
└── src/
    ├── tensor/
    │   ├── engine.h                      # TensorEngine<T> — abstract backend interface
    │   ├── cpu_engine.h                  # CpuTensorEngine<T> — header
    │   ├── cpu_engine.cpp                # CpuTensorEngine<T> — for-loop implementation
    │   ├── tensor.h                      # Tensor<T> — operator overloading + Eigen-like API
    │   └── tensor_benchmark.cpp          # Full benchmark (all types, BLAS L1/L2/L3)
    ├── core/
    └── model.h                       # Model<T> — abstract fit/predict interface
```

Each algorithm is self-contained with its source, benchmark, and example in the same directory.

## Building

### Make

```bash
make all
make clean
```

### CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Tensor Operations

```cpp
CpuTensorEngine<float> e;

Tensor<float> a(e, {1.0f, 2.0f, 3.0f, 4.0f});
Tensor<float> b(e, {5.0f, 6.0f, 7.0f, 8.0f});
Tensor<float> m(e, 2, 2);          // 2x2 matrix

auto c = a + b;                     // [6, 8, 10, 12]
auto d = a * 3.0f;                  // [3, 6, 9, 12]
auto e = 10.0f - a;                 // [9, 8, 7, 6]
auto f = 12.0f / a;                 // [12, 6, 4, 3]

a += b;
a *= 2.0f;
a /= b;

auto g = -a;                        // negation
++a;                                // pre-increment
a++;                                // post-increment

bool all = a < b;                   // element-wise <, returns true if ALL pass
bool any = a[0] < 5.0f;             // scalar comparison

m(0,0) = 1; m(0,1) = 2;
m(1,0) = 3; m(1,1) = 4;

auto mt = m.transpose();            // transpose
auto prod = m.matmul(m);            // matrix multiply
auto inv = m.inv();                  // inverse
float d = m.det();                  // determinant

auto y = m.gemv(x);                 // y = m * x
auto c = m.gemm(b);                 // C = A * B
m.ger(x, y);                        // A += x * y^T

auto I = Tensor<float>::eye(e, 3);          // 3x3 identity
auto z = Tensor<float>::zeros(e, 4, 4);     // 4x4 zeros
auto o = Tensor<float>::ones(e, 2, 3);      // 2x3 ones

auto s = a.sqrt();
auto ex = a.exp();

auto flat = a.reshape(2, 2);
auto col = m.col(0);
auto row = m.row(1);
auto blk = m.block(0, 0, 1, 2);
```

## Adding a New Backend

1. Copy `src/tensor/cpu_engine.h` and `cpu_engine.cpp`
2. Rename to your backend (e.g., `cuda_engine.h`, `opencl_engine.h`)
3. Implement all `TensorEngine<T>` methods with your backend's primitives
4. Swap in your engine — all ML algorithms work unchanged

```cpp
// Your backend
template <typename T>
class CudaTensorEngine : public TensorEngine<T> {
    T* malloc(size_t n) override { /* cudaMalloc */ }
    void gemm(...) override { /* cublasSgemm */ }
    // ... implement all methods
};

// Usage is identical
CudaTensorEngine<float> engine;
Tensor<float> a(engine, {1.0f, 2.0f});
```

## Adding a New ML Algorithm

Create a directory under `src/` following the pattern:

```
src/supervised/classification/
├── logistic_regression.h
├── logistic_regression.cpp
├── logistic_regression_benchmark.cpp
└── logistic_regression_example.cpp
```

Inherit from `Model<T>` and implement `fit()` and `predict()`.

## API Reference

### TensorEngine\<T\> (src/tensor/engine.h)

**BLAS Level 1**: `dot`, `axpy`, `nrm2`, `asum`, `iamax`, `copy`, `swap`, `rot`

**BLAS Level 2**: `gemv`, `ger`, `symv`, `trmv`

**BLAS Level 3**: `gemm`, `syrk`, `symm`, `trmm`, `trsm`

**Element-wise**: `add`, `sub`, `elementwise_mul`, `elementwise_div`, `scale`, `add_scalar`

**Math**: `exp`, `log`, `abs`, `sqrt`, `neg`

**Reductions**: `sum`, `max`, `min`, `argmax`, `argmin`, `sum_axis`, `max_axis`

**Linear algebra**: `transpose`, `inverse`, `determinant`, `eye`

### Tensor\<T\> (src/tensor/tensor.h)

**Operators**: `+`, `-`, `*`, `/` (tensor×tensor, tensor×scalar, scalar×tensor)
`=`, `+=`, `-=`, `*=`, `/=` (compound assignment)
`-`, `+`, `++`, `--` (unary)
`==`, `!=`, `<`, `>`, `<=`, `>=` (comparison)

**BLAS wrappers**: `matmul`, `gemm`, `matvec`, `gemv`, `ger`, `syrk`, `symm`, `trmm`, `nrm2`, `asum`, `iamax`, `copy_to`

**Shape**: `reshape`, `transpose`, `row`, `col`, `block`, `hstack`, `vstack`

**Math**: `exp`, `log`, `abs`, `sqrt`, `neg`

**Factory**: `eye`, `zeros`, `ones`, `constant`

## Compiler Requirements

- C++17 or later
- Tested with GCC 13
