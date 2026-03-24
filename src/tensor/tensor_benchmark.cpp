#include "../tensor/cpu_engine.h"
#include "../tensor/tensor.h"
#include <chrono>
#include <cstdint>
#include <cstdio>

using namespace ml;
using Clock = std::chrono::high_resolution_clock;

static void hline() {
    printf("+------------+------+------------+-----------+-----------+\n");
}

static void hline_wide() {
    printf("+------------+------+------------+-----------+-----------+-----------+\n");
}

template <typename T>
void bench_vector(const char* type_name, size_t size) {
    CpuTensorEngine<T> engine;
    Tensor<T> a(engine, size);
    Tensor<T> b(engine, size);
    Tensor<T> c(engine, size);

    a.fill(T(1));
    b.fill(T(2));

    int iters = 100;

    auto run = [&](const char* op, auto&& fn) {
        fn();
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gops = (static_cast<double>(size) * iters / 1e9) / (ms / 1e3);
        printf("| %-10s | %-4s | %10zu | %9.2f | %9.3f |\n",
               type_name, op, size, ms, gops);
    };

    run("add", [&]() { engine.add(a.data(), b.data(), c.data(), size); });
    run("sub", [&]() { engine.sub(a.data(), b.data(), c.data(), size); });
    run("mul", [&]() { engine.elementwise_mul(a.data(), b.data(), c.data(), size); });
    run("scl", [&]() { engine.scale(a.data(), T(3), c.data(), size); });
    run("neg", [&]() { engine.neg(a.data(), c.data(), size); });
    run("abs", [&]() { engine.abs(a.data(), c.data(), size); });

    T d = T(0);
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) d += engine.dot(a.data(), b.data(), size);
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gops = (static_cast<double>(size) * 2 * iters / 1e9) / (ms / 1e3);
    printf("| %-10s | %-4s | %10zu | %9.2f | %9.3f |\n",
           type_name, "dot", size, ms, gops);

    start = Clock::now();
    for (int i = 0; i < iters; ++i) engine.axpy(T(2), a.data(), c.data(), size);
    end = Clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();
    gops = (static_cast<double>(size) * 2 * iters / 1e9) / (ms / 1e3);
    printf("| %-10s | %-4s | %10zu | %9.2f | %9.3f |\n",
           type_name, "axpy", size, ms, gops);
}

void bench_matrix(size_t m, size_t k, size_t n) {
    CpuTensorEngine<float> engine;

    Tensor<float> A(engine, m, k);
    Tensor<float> B(engine, k, n);
    Tensor<float> C(engine, m, n);
    Tensor<float> v(engine, k);
    A.fill(1.0f);
    B.fill(2.0f);
    v.fill(1.0f);

    int iters = 100;

    auto run_mat = [&](const char* op, auto&& fn) {
        fn();
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * m * k * n * iters / 1e9) / (ms / 1e3);
        printf("| %-10s | %-4s | %4zux%-4zu | %9.2f | %9.3f |\n",
               "float", op, m, n, ms, gflops);
    };

    auto run_matvec = [&](const char* op, auto&& fn) {
        fn();
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * m * k * iters / 1e9) / (ms / 1e3);
        printf("| %-10s | %-4s | %4zux%-4zu | %9.2f | %9.3f |\n",
               "float", op, m, k, ms, gflops);
    };

    run_mat("gemm", [&]() { engine.gemm(Trans::No, Trans::No, m, n, k, 1.0f, A.data(), B.data(), 0.0f, C.data()); });
    run_matvec("gemv", [&]() { engine.gemv(Trans::No, m, k, 1.0f, A.data(), v.data(), 0.0f, C.data()); });
}

void bench_transpose(size_t m, size_t n) {
    CpuTensorEngine<float> engine;
    Tensor<float> A(engine, m, n);
    Tensor<float> B(engine, n, m);
    A.fill(1.0f);

    int iters = 100;
    A.transpose();
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i)
        engine.transpose(A.data(), B.data(), m, n);
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gops = (static_cast<double>(m * n) * iters / 1e9) / (ms / 1e3);

    printf("| %-10s | %-4s | %4zux%-4zu | %9.2f | %9.3f |\n",
           "float", "T", m, n, ms, gops);
}

void bench_inverse_det(size_t n) {
    CpuTensorEngine<float> engine;
    Tensor<float> A = Tensor<float>::eye(engine, n);
    for (size_t i = 0; i < n; ++i)
        A(i, i) = float(n + i + 1);

    int iters = 10;
    Tensor<float> Inv(engine, n, n);

    engine.inverse(A.data(), Inv.data(), n);
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i)
        engine.inverse(A.data(), Inv.data(), n);
    auto end = Clock::now();
    double ms_inv = std::chrono::duration<double, std::milli>(end - start).count() / iters;

    start = Clock::now();
    float d = 0;
    for (int i = 0; i < iters; ++i)
        d += engine.determinant(A.data(), n);
    end = Clock::now();
    double ms_det = std::chrono::duration<double, std::milli>(end - start).count() / iters;

    printf("| %-10s | %-4s | %4zux%-4zu | %9.2f |         - |\n",
           "float", "inv", n, n, ms_inv);
    printf("| %-10s | %-4s | %4zux%-4zu | %9.2f |         - |\n",
           "float", "det", n, n, ms_det);
}

void bench_axis_reductions(size_t rows, size_t cols) {
    CpuTensorEngine<float> engine;
    Tensor<float> A(engine, rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) A[i] = float(i % 100);

    int iters = 100;

    Tensor<float> res0(engine, 1, cols);
    engine.sum_axis(A.data(), res0.data(), rows, cols, 0);
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i)
        engine.sum_axis(A.data(), res0.data(), rows, cols, 0);
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    printf("| %-10s | %-4s | %4zux%-4zu | %9.2f |         - |\n",
           "float", "sum0", rows, cols, ms);

    Tensor<float> res1(engine, rows, 1);
    engine.sum_axis(A.data(), res1.data(), rows, cols, 1);
    start = Clock::now();
    for (int i = 0; i < iters; ++i)
        engine.sum_axis(A.data(), res1.data(), rows, cols, 1);
    end = Clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    printf("| %-10s | %-4s | %4zux%-4zu | %9.2f |         - |\n",
           "float", "sum1", rows, cols, ms);

    engine.max_axis(A.data(), res0.data(), rows, cols, 0);
    start = Clock::now();
    for (int i = 0; i < iters; ++i)
        engine.max_axis(A.data(), res0.data(), rows, cols, 0);
    end = Clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    printf("| %-10s | %-4s | %4zux%-4zu | %9.2f |         - |\n",
           "float", "max0", rows, cols, ms);
}

void bench_exp_log_sqrt(size_t size) {
    CpuTensorEngine<float> engine;
    Tensor<float> a(engine, size);
    Tensor<float> b(engine, size);
    for (size_t i = 0; i < size; ++i) a[i] = float(i + 1) * 0.01f;

    int iters = 100;

    auto run = [&](const char* op, auto&& fn) {
        fn();
        auto start = Clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gops = (static_cast<double>(size) * iters / 1e9) / (ms / 1e3);
        printf("| %-10s | %-4s | %10zu | %9.2f | %9.3f |\n",
               "float", op, size, ms, gops);
    };

    run("exp", [&]() { engine.exp(a.data(), b.data(), size); });
    run("log", [&]() { engine.log(a.data(), b.data(), size); });
    run("sqt", [&]() { engine.sqrt(a.data(), b.data(), size); });
}

int main() {
    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════╗\n");
    printf("  ║          Tensor Engine — Full CPU Benchmark           ║\n");
    printf("  ╚═══════════════════════════════════════════════════════╝\n\n");

    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │  VECTOR OPERATIONS (all types)                            │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    size_t vec_sizes[] = {1024, 65536, 1048576};

    for (size_t sz : vec_sizes) {
        printf("  Vector size: %zu\n\n", sz);
        hline();
        printf("| %-10s | %-4s | %10s | %9s | %9s |\n",
               "Type", "Op", "Size", "Time (ms)", "GOps/s");
        hline();

        bench_vector<float>("float", sz);
        bench_vector<double>("double", sz);
        bench_vector<int8_t>("int8", sz);
        bench_vector<uint8_t>("uint8", sz);
        bench_vector<int16_t>("int16", sz);
        bench_vector<uint16_t>("uint16", sz);
        bench_vector<int32_t>("int32", sz);
        bench_vector<uint32_t>("uint32", sz);
        bench_vector<int64_t>("int64", sz);
        bench_vector<uint64_t>("uint64", sz);

        hline();
        printf("\n");
    }

    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │  ELEMENT-WISE MATH (float)                                │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    hline();
    printf("| %-10s | %-4s | %10s | %9s | %9s |\n",
           "Type", "Op", "Size", "Time (ms)", "GOps/s");
    hline();

    for (size_t sz : vec_sizes)
        bench_exp_log_sqrt(sz);

    hline();
    printf("\n");

    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │  MATRIX OPERATIONS (float)                                │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    hline_wide();
    printf("| %-10s | %-4s | %9s | %9s | %9s | %9s |\n",
           "Type", "Op", "Shape", "Time (ms)", "GFLOP/s", "");
    hline_wide();

    for (size_t s : {256, 512, 1024})
        bench_transpose(s, s);

    bench_matrix(128, 128, 128);
    bench_matrix(256, 256, 256);
    bench_matrix(512, 128, 512);

    for (size_t s : {4, 8, 16, 32, 64, 128})
        bench_inverse_det(s);

    bench_axis_reductions(1000, 100);
    bench_axis_reductions(10000, 50);

    hline_wide();
    printf("\n");

    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │  BLAS LEVEL 1 EXTRA (float)                               │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    hline();
    printf("| %-10s | %-4s | %10s | %9s | %9s |\n",
           "Type", "Op", "Size", "Time (ms)", "GOps/s");
    hline();

    for (size_t sz : vec_sizes) {
        CpuTensorEngine<float> en;
        Tensor<float> a(en, sz), b(en, sz);
        for (size_t i = 0; i < sz; ++i) { a[i] = float(i) * 0.01f; b[i] = float(sz - i) * 0.01f; }
        int iters = 100;

        auto run_l1 = [&](const char* op, auto&& fn) {
            fn();
            auto s = Clock::now();
            for (int i = 0; i < iters; ++i) fn();
            auto e = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(e - s).count();
            double gops = (static_cast<double>(sz) * iters / 1e9) / (ms / 1e3);
            printf("| %-10s | %-4s | %10zu | %9.2f | %9.3f |\n",
                   "float", op, sz, ms, gops);
        };

        run_l1("nrm2", [&]() { en.nrm2(a.data(), sz); });
        run_l1("asum", [&]() { en.asum(a.data(), sz); });
        run_l1("copy", [&]() { en.copy(a.data(), b.data(), sz); });
        run_l1("swap", [&]() { en.swap(a.data(), b.data(), sz); });
        run_l1("rot",  [&]() { en.rot(a.data(), b.data(), sz, 0.707f, 0.707f); });
    }
    hline();
    printf("\n");

    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │  BLAS LEVEL 2 (float)                                     │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    hline_wide();
    printf("| %-10s | %-4s | %9s | %9s | %9s | %9s |\n",
           "Type", "Op", "Shape", "Time (ms)", "GFLOP/s", "");
    hline_wide();

    for (size_t m : {256, 512, 1024}) {
        size_t n = m;
        CpuTensorEngine<float> en;
        Tensor<float> A(en, m, n), x(en, n), y(en, m);
        A.fill(1.0f); x.fill(1.0f); y.fill(0.0f);
        int iters = 100;

        auto run_blas2 = [&](const char* op, auto&& fn, size_t flops) {
            fn();
            auto s = Clock::now();
            for (int i = 0; i < iters; ++i) fn();
            auto e = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(e - s).count();
            double gflops = (static_cast<double>(flops) * iters / 1e9) / (ms / 1e3);
            printf("| %-10s | %-4s | %4zux%-4zu | %9.2f | %9.3f |\n",
                   "float", op, m, n, ms, gflops);
        };

        run_blas2("gemv", [&]() { en.gemv(Trans::No, m, n, 1.0f, A.data(), x.data(), 0.0f, y.data()); }, 2 * m * n);
        run_blas2("ger ", [&]() { en.ger(m, n, 1.0f, x.data(), x.data(), A.data()); }, 2 * m * n);
        run_blas2("symv", [&]() { en.symv(Uplo::Lower, m, 1.0f, A.data(), x.data(), 0.0f, y.data()); }, 2 * m * n);
        run_blas2("trmv", [&]() { en.trmv(Uplo::Lower, Trans::No, m, A.data(), x.data()); }, m * m);
    }
    hline_wide();
    printf("\n");

    printf("  ┌───────────────────────────────────────────────────────────┐\n");
    printf("  │  BLAS LEVEL 3 (float)                                     │\n");
    printf("  └───────────────────────────────────────────────────────────┘\n\n");

    hline_wide();
    printf("| %-10s | %-4s | %9s | %9s | %9s | %9s |\n",
           "Type", "Op", "Shape", "Time (ms)", "GFLOP/s", "");
    hline_wide();

    for (size_t n : {128, 256, 512}) {
        size_t m = n, k = n;
        CpuTensorEngine<float> en;
        Tensor<float> A(en, m, k), B(en, k, n), C(en, m, n);
        A.fill(1.0f); B.fill(1.0f); C.fill(0.0f);
        int iters = 10;

        auto run_blas3 = [&](const char* op, auto&& fn, size_t flops) {
            fn();
            auto s = Clock::now();
            for (int i = 0; i < iters; ++i) fn();
            auto e = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(e - s).count();
            double gflops = (static_cast<double>(flops) * iters / 1e9) / (ms / 1e3);
            printf("| %-10s | %-4s | %4zux%-4zu | %9.2f | %9.3f |\n",
                   "float", op, m, n, ms, gflops);
        };

        run_blas3("gemm", [&]() { en.gemm(Trans::No, Trans::No, m, n, k, 1.0f, A.data(), B.data(), 0.0f, C.data()); }, 2 * m * n * k);

        run_blas3("TA*B", [&]() { en.gemm(Trans::Yes, Trans::No, m, n, k, 1.0f, A.data(), B.data(), 0.0f, C.data()); }, 2 * m * n * k);

        run_blas3("A*TB", [&]() { en.gemm(Trans::No, Trans::Yes, m, n, k, 1.0f, A.data(), B.data(), 0.0f, C.data()); }, 2 * m * n * k);

        run_blas3("syrk", [&]() { en.syrk(Uplo::Lower, m, k, 1.0f, A.data(), 0.0f, C.data()); }, m * m * k);

        run_blas3("trmm", [&]() { en.trmm(Uplo::Lower, Trans::No, m, n, 1.0f, A.data(), B.data()); }, m * m * n);

        Tensor<float> L(en, m, m);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j <= i; ++j)
                L(i, j) = (i == j) ? 2.0f : 0.1f;
        Tensor<float> Bx(en, m, n); Bx.fill(1.0f);
        run_blas3("trsm", [&]() { en.trsm(Uplo::Lower, Trans::No, m, n, 1.0f, L.data(), Bx.data()); }, m * m * n);
    }
    hline_wide();
    printf("\n");

    return 0;
}
