# ML

Dinamik tensör hesaplama arka uçlarına sahip bir C++ makine öğrenimi kütüphanesi. Tensör motoru temiz bir arayüzün arkasında soyutlanmıştır, bu da makine öğrenimi algoritma kodunu değiştirmeden CPU, CUDA, OpenCL veya özel donanım arka uçları arasında geçiş yapmayı kolaylaştırır.

## Özellikler

- **Takılabilir arka uç mimarisi** — `CpuTensorEngine`'i herhangi bir özel motorla değiştirin
- **Tam BLAS Seviye 1–3** — `dot`, `axpy`, `gemv`, `ger`, `gemm`, `syrk`, `trmm`, `trsm`, ...
- **10 veri tipi** — `float`, `double`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64`
- **Eigen benzeri tensör API'si** — operatör işlemleri (`+`, `-`, `*`, `/`, `+=`, `-=`, karşılaştırmalar, artırma/azaltma)
- **Şekil duyarlı (shape-aware) tensörler** — `reshape`, `transpose`, `row`, `col`, `block`, `hstack`, `vstack` ile 1B/2B
- **Eleman bazlı (element-wise) matematik** — `exp`, `log`, `sqrt`, `abs`, `neg`
- **Eksen indirgemeleri (Axis reductions)** — satırlar veya sütunlar boyunca `sum(axis)`, `max(axis)`
- **Doğrusal cebir** — `inverse`, `determinant`, `eye`
- **Karşılaştırmalı testler (Benchmarks)** — Tüm tipler ve BLAS seviyeleri için işlem başına zamanlama

## Proje Yapısı

```
ml/
├── CMakeLists.txt
├── Makefile
└── src/
    ├── tensor/
    │   ├── engine.h                      # TensorEngine<T> — soyut arka uç arayüzü
    │   ├── cpu_engine.h                  # CpuTensorEngine<T> — başlık dosyası
    │   ├── cpu_engine.cpp                # CpuTensorEngine<T> — for-döngüsü uygulaması
    │   ├── tensor.h                      # Tensor<T> — operatör işlemleri + Eigen benzeri API
    │   └── tensor_benchmark.cpp          # Tam benchmark (tüm tipler, BLAS L1/L2/L3)
    ├── core/
    └── model.h                       # Model<T> — soyut fit/predict arayüzü
```

Her algoritma; kaynağı, karşılaştırmalı testi ve örneği aynı dizinde olacak şekilde kendi içinde bağımsızdır.

## Derleme

### Make

```bash
make all                          # Build all
make ALGO=linear_regression       # Build specific algorithm only
make ALGO="linear_regression logistic_regression"  # Build multiple
make linear_regression            # Shorthand for above
make linear_regression_example    # Build only example
make linear_regression_benchmark  # Build only benchmark
make clean
```

### CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Kullanım

### Tensör İşlemleri

```cpp
CpuTensorEngine<float> e;

Tensor<float> a(e, {1.0f, 2.0f, 3.0f, 4.0f});
Tensor<float> b(e, {5.0f, 6.0f, 7.0f, 8.0f});
Tensor<float> m(e, 2, 2);          // 2x2 matris

auto c = a + b;                     // [6, 8, 10, 12]
auto d = a * 3.0f;                  // [3, 6, 9, 12]
auto e = 10.0f - a;                 // [9, 8, 7, 6]
auto f = 12.0f / a;                 // [12, 6, 4, 3]

a += b;
a *= 2.0f;
a /= b;

auto g = -a;                        // negatifini alma
++a;                                // ön-artırma
a++;                                // son-artırma

bool all = a < b;                   // eleman bazlı <, TÜMÜ geçerse true döndürür
bool any = a[0] < 5.0f;             // skaler karşılaştırma

m(0,0) = 1; m(0,1) = 2;
m(1,0) = 3; m(1,1) = 4;

auto mt = m.transpose();            // transpoz
auto prod = m.matmul(m);            // matris çarpımı
auto inv = m.inv();                 // ters (inverse) matris
float d = m.det();                  // determinant

auto y = m.gemv(x);                 // y = m * x
auto c = m.gemm(b);                 // C = A * B
m.ger(x, y);                        // A += x * y^T

auto I = Tensor<float>::eye(e, 3);          // 3x3 birim matris
auto z = Tensor<float>::zeros(e, 4, 4);     // 4x4 sıfırlar
auto o = Tensor<float>::ones(e, 2, 3);      // 2x3 birler

auto s = a.sqrt();
auto ex = a.exp();

auto flat = a.reshape(2, 2);
auto col = m.col(0);
auto row = m.row(1);
auto blk = m.block(0, 0, 1, 2);
```

## Yeni Bir Arka Uç Eklemek

1. `src/tensor/cpu_engine.h` ve `cpu_engine.cpp` dosyalarını kopyalayın
2. Arka ucunuza göre yeniden adlandırın (ör. `cuda_engine.h`, `opencl_engine.h`)
3. Tüm `TensorEngine<T>` metotlarını kendi arka ucunuzun temel yapılarıyla (primitives) uygulayın
4. Motorunuzu değiştirip kullanın — tüm makine öğrenimi algoritmaları değişmeden çalışır

```cpp
// Sizin arka ucunuz
template <typename T>
class CudaTensorEngine : public TensorEngine<T> {
    T* malloc(size_t n) override { /* cudaMalloc */ }
    void gemm(...) override { /* cublasSgemm */ }
    // ... tüm metotları uygulayın
};

// Kullanım tamamen aynıdır
CudaTensorEngine<float> engine;
Tensor<float> a(engine, {1.0f, 2.0f});
```

## Yeni Bir ML Algoritması Eklemek

`src/` altında şu deseni izleyerek bir dizin oluşturun:

```
src/supervised/classification/
├── logistic_regression.h
├── logistic_regression.cpp
├── logistic_regression_benchmark.cpp
└── logistic_regression_example.cpp
```

`Model<T>` sınıfından kalıtım alın (inherit) ve `fit()` ile `predict()` metotlarını uygulayın.

## API Referansı

### TensorEngine\<T\> (src/tensor/engine.h)

**BLAS Seviye 1**: `dot`, `axpy`, `nrm2`, `asum`, `iamax`, `copy`, `swap`, `rot`

**BLAS Seviye 2**: `gemv`, `ger`, `symv`, `trmv`

**BLAS Seviye 3**: `gemm`, `syrk`, `symm`, `trmm`, `trsm`

**Eleman bazlı**: `add`, `sub`, `elementwise_mul`, `elementwise_div`, `scale`, `add_scalar`

**Matematik**: `exp`, `log`, `abs`, `sqrt`, `neg`

**İndirgemeler**: `sum`, `max`, `min`, `argmax`, `argmin`, `sum_axis`, `max_axis`

**Doğrusal cebir**: `transpose`, `inverse`, `determinant`, `eye`

### Tensor\<T\> (src/tensor/tensor.h)

**Operatörler**: `+`, `-`, `*`, `/` (tensör×tensör, tensör×skaler, skaler×tensör)
`=`, `+=`, `-=`, `*=`, `/=` (bileşik atama)
`-`, `+`, `++`, `--` (tekil/unary)
`==`, `!=`, `<`, `>`, `<=`, `>=` (karşılaştırma)

**BLAS sarmalayıcıları (wrappers)**: `matmul`, `gemm`, `matvec`, `gemv`, `ger`, `syrk`, `symm`, `trmm`, `nrm2`, `asum`, `iamax`, `copy_to`

**Şekil (Shape)**: `reshape`, `transpose`, `row`, `col`, `block`, `hstack`, `vstack`

**Matematik**: `exp`, `log`, `abs`, `sqrt`, `neg`

**Üretici (Factory)**: `eye`, `zeros`, `ones`, `constant`

## Derleyici Gereksinimleri

- C++17 veya daha yenisi
- GCC 13 ile test edilmiştir
