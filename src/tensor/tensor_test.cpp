#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "tensor/tensor.h"
#include "tensor/cpu_engine.h"

using namespace ml;

static constexpr float kTol = 1e-5f;

#define EXPECT_TENSOR_EQ(a, b)                              \
    do {                                                    \
        ASSERT_EQ((a).size(), (b).size());                  \
        ASSERT_EQ((a).rows(), (b).rows());                  \
        ASSERT_EQ((a).cols(), (b).cols());                  \
        for (size_t _i = 0; _i < (a).size(); ++_i)         \
            EXPECT_NEAR((a)[_i], (b)[_i], kTol);           \
    } while (0)

class TensorTest : public ::testing::Test {
protected:
    CpuTensorEngine<float> eng;

    Tensor<float> make(std::initializer_list<float> init) {
        return Tensor<float>(eng, init);
    }

    Tensor<float> make2d(size_t rows, size_t cols, std::initializer_list<float> init) {
        Tensor<float> t(eng, rows, cols);
        size_t i = 0;
        for (auto v : init) t[i++] = v;
        return t;
    }
};

TEST_F(TensorTest, Construct1D) {
    Tensor<float> t(eng, 5);
    EXPECT_EQ(t.size(), 5u);
    EXPECT_EQ(t.rows(), 1u);
    EXPECT_EQ(t.cols(), 5u);
    EXPECT_TRUE(t.is_vector());
    EXPECT_FALSE(t.is_matrix());
}

TEST_F(TensorTest, Construct2D) {
    Tensor<float> t(eng, 3, 4);
    EXPECT_EQ(t.size(), 12u);
    EXPECT_EQ(t.rows(), 3u);
    EXPECT_EQ(t.cols(), 4u);
    EXPECT_TRUE(t.is_matrix());
    EXPECT_FALSE(t.is_vector());
}

TEST_F(TensorTest, ConstructInitializerList) {
    Tensor<float> t(eng, {1.f, 2.f, 3.f});
    EXPECT_EQ(t.size(), 3u);
    EXPECT_FLOAT_EQ(t[0], 1.f);
    EXPECT_FLOAT_EQ(t[1], 2.f);
    EXPECT_FLOAT_EQ(t[2], 3.f);
}

TEST_F(TensorTest, ConstructFromVector) {
    std::vector<float> v = {4.f, 5.f, 6.f};
    Tensor<float> t(eng, v);
    EXPECT_EQ(t.size(), 3u);
    EXPECT_FLOAT_EQ(t[0], 4.f);
    EXPECT_FLOAT_EQ(t[1], 5.f);
    EXPECT_FLOAT_EQ(t[2], 6.f);
}

TEST_F(TensorTest, ConstructFromRawPointer1DNonOwning) {
    float buf[] = {10.f, 20.f};
    Tensor<float> t(eng, buf, 2, false);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_FLOAT_EQ(t[0], 10.f);
    EXPECT_FLOAT_EQ(t[1], 20.f);
}

TEST_F(TensorTest, ConstructFromRawPointer1DOwning) {
    float* buf = eng.malloc(3);
    buf[0] = 1.f; buf[1] = 2.f; buf[2] = 3.f;
    Tensor<float> t(eng, buf, 3, true);
    EXPECT_EQ(t.size(), 3u);
    EXPECT_FLOAT_EQ(t[1], 2.f);
}

TEST_F(TensorTest, ConstructFromRawPointer2DNonOwning) {
    float buf[] = {1.f, 2.f, 3.f, 4.f};
    Tensor<float> t(eng, buf, 2, 2, false);
    EXPECT_EQ(t.rows(), 2u);
    EXPECT_EQ(t.cols(), 2u);
    EXPECT_FLOAT_EQ(t(1, 0), 3.f);
}

TEST_F(TensorTest, ConstructFromRawPointer2DOwning) {
    float* buf = eng.malloc(4);
    buf[0] = 1; buf[1] = 2; buf[2] = 3; buf[3] = 4;
    Tensor<float> t(eng, buf, 2, 2, true);
    EXPECT_FLOAT_EQ(t(0, 1), 2.f);
}

TEST_F(TensorTest, CopyConstructor) {
    Tensor<float> a = make({1.f, 2.f, 3.f});
    Tensor<float> b(a);
    EXPECT_EQ(b.size(), 3u);
    EXPECT_NE(b.data(), a.data());
    EXPECT_TENSOR_EQ(a, b);
}

TEST_F(TensorTest, MoveConstructor) {
    Tensor<float> a = make({1.f, 2.f, 3.f});
    float* ptr = a.data();
    Tensor<float> b(std::move(a));
    EXPECT_EQ(b.size(), 3u);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
}

TEST_F(TensorTest, CopyAssignment) {
    Tensor<float> a = make({1.f, 2.f});
    Tensor<float> b(eng, 5);
    b = a;
    EXPECT_EQ(b.size(), 2u);
    EXPECT_TENSOR_EQ(a, b);
}

TEST_F(TensorTest, CopyAssignmentSelf) {
    Tensor<float> a = make({1.f, 2.f, 3.f});
    a = a;
    EXPECT_EQ(a.size(), 3u);
    EXPECT_FLOAT_EQ(a[0], 1.f);
}

TEST_F(TensorTest, MoveAssignment) {
    Tensor<float> a = make({4.f, 5.f});
    Tensor<float> b(eng, 5);
    float* ptr = a.data();
    b = std::move(a);
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(a.data(), nullptr);
}

TEST_F(TensorTest, MoveAssignmentSelf) {
    Tensor<float> a = make({7.f});
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
    a = std::move(a);
#pragma GCC diagnostic pop
    EXPECT_FLOAT_EQ(a[0], 7.f);
}

TEST_F(TensorTest, Accessors) {
    Tensor<float> t(eng, 3, 4);
    EXPECT_EQ(t.size(), 12u);
    EXPECT_EQ(t.rows(), 3u);
    EXPECT_EQ(t.cols(), 4u);
    EXPECT_TRUE(t.is_matrix());
    EXPECT_FALSE(t.is_vector());
    Tensor<float> v(eng, 5);
    EXPECT_TRUE(v.is_vector());
    EXPECT_FALSE(v.is_matrix());
}

TEST_F(TensorTest, DataMutable) {
    Tensor<float> t = make({1.f, 2.f});
    t.data()[0] = 99.f;
    EXPECT_FLOAT_EQ(t[0], 99.f);
}

TEST_F(TensorTest, DataConst) {
    Tensor<float> t = make({1.f, 2.f});
    const Tensor<float>& ct = t;
    const float* p = ct.data();
    EXPECT_FLOAT_EQ(p[0], 1.f);
}

TEST_F(TensorTest, EngineRef) {
    Tensor<float> t(eng, 3);
    TensorEngine<float>& e = t.engine();
    EXPECT_NE(&e, nullptr);
}

TEST_F(TensorTest, OperatorBracketMutable) {
    Tensor<float> t = make({1.f, 2.f, 3.f});
    t[1] = 10.f;
    EXPECT_FLOAT_EQ(t[1], 10.f);
}

TEST_F(TensorTest, OperatorBracketConst) {
    Tensor<float> t = make({1.f, 2.f, 3.f});
    const Tensor<float>& ct = t;
    EXPECT_FLOAT_EQ(ct[2], 3.f);
}

TEST_F(TensorTest, OperatorParenMutable) {
    Tensor<float> t = make2d(2, 3, {1, 2, 3, 4, 5, 6});
    t(1, 2) = 99.f;
    EXPECT_FLOAT_EQ(t(1, 2), 99.f);
}

TEST_F(TensorTest, OperatorParenConst) {
    Tensor<float> t = make2d(2, 3, {1, 2, 3, 4, 5, 6});
    const Tensor<float>& ct = t;
    EXPECT_FLOAT_EQ(ct(0, 2), 3.f);
    EXPECT_FLOAT_EQ(ct(1, 0), 4.f);
}

TEST_F(TensorTest, Reshape) {
    Tensor<float> t(eng, 6);
    t.fill(1.f);
    Tensor<float> r = t.reshape(2, 3);
    EXPECT_EQ(r.rows(), 2u);
    EXPECT_EQ(r.cols(), 3u);
    EXPECT_EQ(r.data(), t.data());
}

TEST_F(TensorTest, ReshapeThrows) {
    Tensor<float> t(eng, 5);
    EXPECT_THROW(t.reshape(2, 3), std::invalid_argument);
}

TEST_F(TensorTest, Transpose1D) {
    Tensor<float> t = make({1.f, 2.f, 3.f});
    Tensor<float> r = t.transpose();
    EXPECT_EQ(r.rows(), 3u);
    EXPECT_EQ(r.cols(), 1u);
}

TEST_F(TensorTest, Transpose2D) {
    Tensor<float> t = make2d(2, 3, {1, 2, 3, 4, 5, 6});
    Tensor<float> r = t.transpose();
    EXPECT_EQ(r.rows(), 3u);
    EXPECT_EQ(r.cols(), 2u);
    EXPECT_FLOAT_EQ(r(0, 0), 1.f);
    EXPECT_FLOAT_EQ(r(2, 1), 6.f);
    EXPECT_FLOAT_EQ(r(1, 0), 2.f);
    EXPECT_FLOAT_EQ(r(0, 1), 4.f);
}

TEST_F(TensorTest, Row) {
    Tensor<float> t = make2d(3, 2, {1, 2, 3, 4, 5, 6});
    Tensor<float> r = t.row(1);
    EXPECT_EQ(r.size(), 2u);
    EXPECT_FLOAT_EQ(r[0], 3.f);
    EXPECT_FLOAT_EQ(r[1], 4.f);
}

TEST_F(TensorTest, Col) {
    Tensor<float> t = make2d(3, 2, {1, 2, 3, 4, 5, 6});
    Tensor<float> c = t.col(1);
    EXPECT_EQ(c.size(), 3u);
    EXPECT_FLOAT_EQ(c[0], 2.f);
    EXPECT_FLOAT_EQ(c[1], 4.f);
    EXPECT_FLOAT_EQ(c[2], 6.f);
}

TEST_F(TensorTest, Block) {
    Tensor<float> t = make2d(3, 3, {1,2,3, 4,5,6, 7,8,9});
    Tensor<float> b = t.block(1, 1, 2, 2);
    EXPECT_EQ(b.rows(), 2u);
    EXPECT_EQ(b.cols(), 2u);
    EXPECT_FLOAT_EQ(b(0, 0), 5.f);
    EXPECT_FLOAT_EQ(b(1, 1), 9.f);
}

TEST_F(TensorTest, BlockThrows) {
    Tensor<float> t = make2d(3, 3, {1,2,3, 4,5,6, 7,8,9});
    EXPECT_THROW(t.block(2, 2, 2, 2), std::invalid_argument);
}

TEST_F(TensorTest, HStack) {
    Tensor<float> a = make2d(2, 2, {1, 2, 3, 4});
    Tensor<float> b = make2d(2, 1, {5, 6});
    Tensor<float> r = Tensor<float>::hstack(a, b);
    EXPECT_EQ(r.rows(), 2u);
    EXPECT_EQ(r.cols(), 3u);
    EXPECT_FLOAT_EQ(r(0, 2), 5.f);
    EXPECT_FLOAT_EQ(r(1, 2), 6.f);
}

TEST_F(TensorTest, HStackThrows) {
    Tensor<float> a = make2d(2, 2, {1, 2, 3, 4});
    Tensor<float> b = make2d(3, 1, {5, 6, 7});
    EXPECT_THROW(Tensor<float>::hstack(a, b), std::invalid_argument);
}

TEST_F(TensorTest, VStack) {
    Tensor<float> a = make2d(1, 3, {1, 2, 3});
    Tensor<float> b = make2d(2, 3, {4, 5, 6, 7, 8, 9});
    Tensor<float> r = Tensor<float>::vstack(a, b);
    EXPECT_EQ(r.rows(), 3u);
    EXPECT_EQ(r.cols(), 3u);
    EXPECT_FLOAT_EQ(r(2, 0), 7.f);
}

TEST_F(TensorTest, VStackThrows) {
    Tensor<float> a = make2d(1, 3, {1, 2, 3});
    Tensor<float> b = make2d(2, 4, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_THROW(Tensor<float>::vstack(a, b), std::invalid_argument);
}

TEST_F(TensorTest, AddTensor) {
    Tensor<float> a = make({1, 2, 3});
    Tensor<float> b = make({4, 5, 6});
    Tensor<float> r = a + b;
    EXPECT_FLOAT_EQ(r[0], 5.f);
    EXPECT_FLOAT_EQ(r[1], 7.f);
    EXPECT_FLOAT_EQ(r[2], 9.f);
}

TEST_F(TensorTest, SubTensor) {
    Tensor<float> a = make({10, 20, 30});
    Tensor<float> b = make({1, 2, 3});
    Tensor<float> r = a - b;
    EXPECT_FLOAT_EQ(r[0], 9.f);
    EXPECT_FLOAT_EQ(r[1], 18.f);
    EXPECT_FLOAT_EQ(r[2], 27.f);
}

TEST_F(TensorTest, MulTensor) {
    Tensor<float> a = make({2, 3, 4});
    Tensor<float> b = make({5, 6, 7});
    Tensor<float> r = a * b;
    EXPECT_FLOAT_EQ(r[0], 10.f);
    EXPECT_FLOAT_EQ(r[1], 18.f);
    EXPECT_FLOAT_EQ(r[2], 28.f);
}

TEST_F(TensorTest, DivTensor) {
    Tensor<float> a = make({10, 20, 30});
    Tensor<float> b = make({2, 4, 5});
    Tensor<float> r = a / b;
    EXPECT_FLOAT_EQ(r[0], 5.f);
    EXPECT_FLOAT_EQ(r[1], 5.f);
    EXPECT_FLOAT_EQ(r[2], 6.f);
}

TEST_F(TensorTest, AddTensorSizeMismatch) {
    Tensor<float> a = make({1, 2});
    Tensor<float> b = make({1, 2, 3});
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST_F(TensorTest, SubTensorSizeMismatch) {
    EXPECT_THROW(make({1}) - make({1, 2}), std::invalid_argument);
}

TEST_F(TensorTest, MulTensorSizeMismatch) {
    EXPECT_THROW(make({1}) * make({1, 2}), std::invalid_argument);
}

TEST_F(TensorTest, DivTensorSizeMismatch) {
    EXPECT_THROW(make({1}) / make({1, 2}), std::invalid_argument);
}

TEST_F(TensorTest, AddScalar) {
    Tensor<float> r = make({1, 2, 3}) + 10.f;
    EXPECT_FLOAT_EQ(r[0], 11.f);
    EXPECT_FLOAT_EQ(r[1], 12.f);
    EXPECT_FLOAT_EQ(r[2], 13.f);
}

TEST_F(TensorTest, SubScalar) {
    Tensor<float> r = make({10, 20}) - 5.f;
    EXPECT_FLOAT_EQ(r[0], 5.f);
    EXPECT_FLOAT_EQ(r[1], 15.f);
}

TEST_F(TensorTest, MulScalar) {
    Tensor<float> r = make({1, 2, 3}) * 3.f;
    EXPECT_FLOAT_EQ(r[0], 3.f);
    EXPECT_FLOAT_EQ(r[1], 6.f);
    EXPECT_FLOAT_EQ(r[2], 9.f);
}

TEST_F(TensorTest, DivScalar) {
    Tensor<float> r = make({10, 20, 30}) / 10.f;
    EXPECT_FLOAT_EQ(r[0], 1.f);
    EXPECT_FLOAT_EQ(r[1], 2.f);
    EXPECT_FLOAT_EQ(r[2], 3.f);
}

TEST_F(TensorTest, ScalarPlusTensor) {
    Tensor<float> r = 10.f + make({1, 2, 3});
    EXPECT_FLOAT_EQ(r[0], 11.f);
    EXPECT_FLOAT_EQ(r[2], 13.f);
}

TEST_F(TensorTest, ScalarMinusTensor) {
    Tensor<float> r = 10.f - make({1, 2, 3});
    EXPECT_FLOAT_EQ(r[0], 9.f);
    EXPECT_FLOAT_EQ(r[1], 8.f);
    EXPECT_FLOAT_EQ(r[2], 7.f);
}

TEST_F(TensorTest, ScalarMulTensor) {
    Tensor<float> r = 3.f * make({1, 2, 3});
    EXPECT_FLOAT_EQ(r[0], 3.f);
    EXPECT_FLOAT_EQ(r[1], 6.f);
    EXPECT_FLOAT_EQ(r[2], 9.f);
}

TEST_F(TensorTest, ScalarDivTensor) {
    Tensor<float> r = 12.f / make({1, 2, 3});
    EXPECT_FLOAT_EQ(r[0], 12.f);
    EXPECT_FLOAT_EQ(r[1], 6.f);
    EXPECT_FLOAT_EQ(r[2], 4.f);
}

TEST_F(TensorTest, AddAssignTensor) {
    Tensor<float> a = make({1, 2, 3});
    a += make({10, 20, 30});
    EXPECT_FLOAT_EQ(a[0], 11.f);
    EXPECT_FLOAT_EQ(a[1], 22.f);
    EXPECT_FLOAT_EQ(a[2], 33.f);
}

TEST_F(TensorTest, SubAssignTensor) {
    Tensor<float> a = make({10, 20, 30});
    a -= make({1, 2, 3});
    EXPECT_FLOAT_EQ(a[0], 9.f);
    EXPECT_FLOAT_EQ(a[1], 18.f);
    EXPECT_FLOAT_EQ(a[2], 27.f);
}

TEST_F(TensorTest, MulAssignTensor) {
    Tensor<float> a = make({2, 3, 4});
    a *= make({5, 6, 7});
    EXPECT_FLOAT_EQ(a[0], 10.f);
    EXPECT_FLOAT_EQ(a[1], 18.f);
    EXPECT_FLOAT_EQ(a[2], 28.f);
}

TEST_F(TensorTest, DivAssignTensor) {
    Tensor<float> a = make({10, 20, 30});
    a /= make({2, 4, 5});
    EXPECT_FLOAT_EQ(a[0], 5.f);
    EXPECT_FLOAT_EQ(a[1], 5.f);
    EXPECT_FLOAT_EQ(a[2], 6.f);
}

TEST_F(TensorTest, AddAssignTensorSizeMismatch) {
    Tensor<float> a = make({1, 2});
    EXPECT_THROW(a += make({1, 2, 3}), std::invalid_argument);
}

TEST_F(TensorTest, SubAssignTensorSizeMismatch) {
    EXPECT_THROW((make({1}) -= make({1, 2})), std::invalid_argument);
}

TEST_F(TensorTest, MulAssignTensorSizeMismatch) {
    EXPECT_THROW((make({1}) *= make({1, 2})), std::invalid_argument);
}

TEST_F(TensorTest, DivAssignTensorSizeMismatch) {
    EXPECT_THROW((make({1}) /= make({1, 2})), std::invalid_argument);
}

TEST_F(TensorTest, AddAssignScalar) {
    Tensor<float> a = make({1, 2, 3});
    a += 10.f;
    EXPECT_FLOAT_EQ(a[0], 11.f);
    EXPECT_FLOAT_EQ(a[2], 13.f);
}

TEST_F(TensorTest, SubAssignScalar) {
    Tensor<float> a = make({10, 20});
    a -= 5.f;
    EXPECT_FLOAT_EQ(a[0], 5.f);
    EXPECT_FLOAT_EQ(a[1], 15.f);
}

TEST_F(TensorTest, MulAssignScalar) {
    Tensor<float> a = make({2, 3});
    a *= 4.f;
    EXPECT_FLOAT_EQ(a[0], 8.f);
    EXPECT_FLOAT_EQ(a[1], 12.f);
}

TEST_F(TensorTest, DivAssignScalar) {
    Tensor<float> a = make({10, 20});
    a /= 10.f;
    EXPECT_FLOAT_EQ(a[0], 1.f);
    EXPECT_FLOAT_EQ(a[1], 2.f);
}

TEST_F(TensorTest, UnaryNegate) {
    Tensor<float> a = make({1, -2, 3});
    Tensor<float> r = -a;
    EXPECT_FLOAT_EQ(r[0], -1.f);
    EXPECT_FLOAT_EQ(r[1], 2.f);
    EXPECT_FLOAT_EQ(r[2], -3.f);
}

TEST_F(TensorTest, UnaryPlus) {
    Tensor<float> a = make({1, 2, 3});
    Tensor<float> r = +a;
    EXPECT_TENSOR_EQ(a, r);
}

TEST_F(TensorTest, PreIncrement) {
    Tensor<float> a = make({0, 1, 2});
    Tensor<float>& r = ++a;
    EXPECT_FLOAT_EQ(a[0], 1.f);
    EXPECT_FLOAT_EQ(a[2], 3.f);
    EXPECT_EQ(&r, &a);
}

TEST_F(TensorTest, PostIncrement) {
    Tensor<float> a = make({0, 1, 2});
    Tensor<float> old = a++;
    EXPECT_FLOAT_EQ(old[0], 0.f);
    EXPECT_FLOAT_EQ(a[0], 1.f);
    EXPECT_FLOAT_EQ(a[2], 3.f);
}

TEST_F(TensorTest, PreDecrement) {
    Tensor<float> a = make({5, 6});
    Tensor<float>& r = --a;
    EXPECT_FLOAT_EQ(a[0], 4.f);
    EXPECT_FLOAT_EQ(a[1], 5.f);
    EXPECT_EQ(&r, &a);
}

TEST_F(TensorTest, PostDecrement) {
    Tensor<float> a = make({5, 6});
    Tensor<float> old = a--;
    EXPECT_FLOAT_EQ(old[0], 5.f);
    EXPECT_FLOAT_EQ(a[0], 4.f);
}

TEST_F(TensorTest, EqualTrue) {
    EXPECT_TRUE(make({1, 2, 3}) == make({1, 2, 3}));
}

TEST_F(TensorTest, EqualFalse) {
    EXPECT_FALSE(make({1, 2, 3}) == make({1, 2, 4}));
}

TEST_F(TensorTest, EqualDifferentSize) {
    EXPECT_FALSE(make({1, 2}) == make({1, 2, 3}));
}

TEST_F(TensorTest, NotEqualTrue) {
    EXPECT_TRUE(make({1, 2}) != make({1, 3}));
}

TEST_F(TensorTest, NotEqualFalse) {
    EXPECT_FALSE(make({1, 2}) != make({1, 2}));
}

TEST_F(TensorTest, LessThanAllTrue) {
    EXPECT_TRUE(make({1, 2, 3}) < make({2, 3, 4}));
}

TEST_F(TensorTest, LessThanFalse) {
    EXPECT_FALSE(make({1, 2, 3}) < make({1, 3, 4}));
}

TEST_F(TensorTest, LessThanSizeMismatch) {
    EXPECT_THROW(make({1}) < make({1, 2}), std::invalid_argument);
}

TEST_F(TensorTest, GreaterThanAllTrue) {
    EXPECT_TRUE(make({3, 4, 5}) > make({2, 3, 4}));
}

TEST_F(TensorTest, GreaterThanFalse) {
    EXPECT_FALSE(make({3, 4, 5}) > make({3, 3, 4}));
}

TEST_F(TensorTest, GreaterThanSizeMismatch) {
    EXPECT_THROW(make({1}) > make({1, 2}), std::invalid_argument);
}

TEST_F(TensorTest, LessEqualTrue) {
    EXPECT_TRUE(make({1, 2, 2}) <= make({1, 2, 3}));
}

TEST_F(TensorTest, LessEqualEqual) {
    EXPECT_TRUE(make({1, 2}) <= make({1, 2}));
}

TEST_F(TensorTest, LessEqualFalse) {
    EXPECT_FALSE(make({1, 3}) <= make({1, 2}));
}

TEST_F(TensorTest, LessEqualSizeMismatch) {
    EXPECT_THROW(make({1}) <= make({1, 2}), std::invalid_argument);
}

TEST_F(TensorTest, GreaterEqualTrue) {
    EXPECT_TRUE(make({3, 4}) >= make({2, 3}));
}

TEST_F(TensorTest, GreaterEqualEqual) {
    EXPECT_TRUE(make({3, 4}) >= make({3, 4}));
}

TEST_F(TensorTest, GreaterEqualFalse) {
    EXPECT_FALSE(make({3, 4}) >= make({4, 4}));
}

TEST_F(TensorTest, GreaterEqualSizeMismatch) {
    EXPECT_THROW(make({1}) >= make({1, 2}), std::invalid_argument);
}

TEST_F(TensorTest, LessScalarTrue) {
    EXPECT_TRUE(make({1, 2, 3}) < 5.f);
}

TEST_F(TensorTest, LessScalarFalse) {
    EXPECT_FALSE(make({1, 2, 5}) < 5.f);
}

TEST_F(TensorTest, GreaterScalarTrue) {
    EXPECT_TRUE(make({5, 6, 7}) > 3.f);
}

TEST_F(TensorTest, GreaterScalarFalse) {
    EXPECT_FALSE(make({5, 6, 3}) > 3.f);
}

TEST_F(TensorTest, LessEqualScalarTrue) {
    EXPECT_TRUE(make({1, 2, 5}) <= 5.f);
}

TEST_F(TensorTest, LessEqualScalarFalse) {
    EXPECT_FALSE(make({1, 2, 6}) <= 5.f);
}

TEST_F(TensorTest, GreaterEqualScalarTrue) {
    EXPECT_TRUE(make({5, 6, 7}) >= 5.f);
}

TEST_F(TensorTest, GreaterEqualScalarFalse) {
    EXPECT_FALSE(make({4, 6, 7}) >= 5.f);
}

TEST_F(TensorTest, Matmul) {
    Tensor<float> A = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> B = make2d(3, 2, {1,2, 3,4, 5,6});
    Tensor<float> C = A.matmul(B);
    EXPECT_EQ(C.rows(), 2u);
    EXPECT_EQ(C.cols(), 2u);
    EXPECT_FLOAT_EQ(C(0, 0), 22.f);
    EXPECT_FLOAT_EQ(C(0, 1), 28.f);
    EXPECT_FLOAT_EQ(C(1, 0), 49.f);
    EXPECT_FLOAT_EQ(C(1, 1), 64.f);
}

TEST_F(TensorTest, MatmulThrows) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    Tensor<float> B = make2d(2, 2, {1,2,3,4});
    EXPECT_THROW(A.matmul(B), std::invalid_argument);
}

TEST_F(TensorTest, Gemm) {
    Tensor<float> A = make2d(2, 2, {1, 2, 3, 4});
    Tensor<float> B = make2d(2, 2, {5, 6, 7, 8});
    Tensor<float> C = A.gemm(B, 2.f, 0.f);
    EXPECT_FLOAT_EQ(C(0, 0), 2.f * (1*5 + 2*7));
    EXPECT_FLOAT_EQ(C(0, 1), 2.f * (1*6 + 2*8));
}

TEST_F(TensorTest, GemmThrows) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    Tensor<float> B = make2d(2, 2, {1,2,3,4});
    EXPECT_THROW(A.gemm(B), std::invalid_argument);
}

TEST_F(TensorTest, Matvec) {
    Tensor<float> A = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> x = make({1, 2, 3});
    Tensor<float> y = A.matvec(x);
    EXPECT_EQ(y.size(), 2u);
    EXPECT_FLOAT_EQ(y[0], 14.f);
    EXPECT_FLOAT_EQ(y[1], 32.f);
}

TEST_F(TensorTest, MatvecThrows) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    Tensor<float> x = make({1, 2});
    EXPECT_THROW(A.matvec(x), std::invalid_argument);
}

TEST_F(TensorTest, Gemv) {
    Tensor<float> A = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> x = make({1, 2, 3});
    Tensor<float> y = A.gemv(x, 2.f, 0.f);
    EXPECT_FLOAT_EQ(y[0], 28.f);
    EXPECT_FLOAT_EQ(y[1], 64.f);
}

TEST_F(TensorTest, GemvThrows) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    EXPECT_THROW(A.gemv(make({1, 2})), std::invalid_argument);
}

TEST_F(TensorTest, Ger) {
    Tensor<float> A = make2d(2, 3, {0,0,0, 0,0,0});
    float xdata[] = {1, 2};
    float ydata[] = {1, 2, 3};
    Tensor<float> x(eng, xdata, 2, 1, false);
    Tensor<float> y(eng, ydata, 3, 1, false);
    A.ger(x, y, 1.f);
    EXPECT_FLOAT_EQ(A(0, 0), 1.f);
    EXPECT_FLOAT_EQ(A(0, 2), 3.f);
    EXPECT_FLOAT_EQ(A(1, 0), 2.f);
    EXPECT_FLOAT_EQ(A(1, 2), 6.f);
}

TEST_F(TensorTest, GerThrows) {
    Tensor<float> A = make2d(2, 3, {0,0,0,0,0,0});
    EXPECT_THROW(A.ger(make({1,2,3}), make({1,2})), std::invalid_argument);
}

TEST_F(TensorTest, Syrk) {
    Tensor<float> A = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> C = A.syrk();
    EXPECT_EQ(C.rows(), 2u);
    EXPECT_EQ(C.cols(), 2u);
    EXPECT_FLOAT_EQ(C(0, 0), 14.f);
    EXPECT_FLOAT_EQ(C(1, 1), 77.f);
    EXPECT_FLOAT_EQ(C(1, 0), 32.f);
}

TEST_F(TensorTest, Symm) {
    Tensor<float> A = make2d(2, 2, {1, 2, 2, 5});
    Tensor<float> B = make2d(2, 2, {1, 0, 0, 1});
    Tensor<float> C = A.symm(B);
    EXPECT_FLOAT_EQ(C(0, 0), 1.f);
    EXPECT_FLOAT_EQ(C(0, 1), 2.f);
    EXPECT_FLOAT_EQ(C(1, 0), 2.f);
    EXPECT_FLOAT_EQ(C(1, 1), 5.f);
}

TEST_F(TensorTest, SymmThrows) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    Tensor<float> B = make2d(3, 2, {1,2,3,4,5,6});
    EXPECT_THROW(A.symm(B), std::invalid_argument);
}

TEST_F(TensorTest, Trmm) {
    Tensor<float> A = make2d(2, 2, {2, 0, 3, 4});
    Tensor<float> B = make2d(2, 2, {1, 2, 3, 4});
    Tensor<float> C = A.trmm(B);
    EXPECT_FLOAT_EQ(C(0, 0), 2.f);
    EXPECT_FLOAT_EQ(C(1, 0), 15.f);
    EXPECT_FLOAT_EQ(C(1, 1), 22.f);
}

TEST_F(TensorTest, TrmmThrows) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    Tensor<float> B = make2d(3, 2, {1,2,3,4,5,6});
    EXPECT_THROW(A.trmm(B), std::invalid_argument);
}

TEST_F(TensorTest, Inv2x2) {
    Tensor<float> A = make2d(2, 2, {1, 2, 3, 4});
    Tensor<float> I = A.inv();
    EXPECT_EQ(I.rows(), 2u);
    EXPECT_NEAR(I(0, 0), -2.0, 1e-4);
    EXPECT_NEAR(I(0, 1),  1.0, 1e-4);
    EXPECT_NEAR(I(1, 0),  1.5, 1e-4);
    EXPECT_NEAR(I(1, 1), -0.5, 1e-4);
}

TEST_F(TensorTest, InvIdentity) {
    Tensor<float> I = Tensor<float>::eye(eng, 3);
    Tensor<float> R = I.inv();
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(R(i, j), (i == j) ? 1.f : 0.f, 1e-5);
}

TEST_F(TensorTest, InvThrowsNonSquare) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    EXPECT_THROW(A.inv(), std::invalid_argument);
}

TEST_F(TensorTest, InvThrowsSingular) {
    Tensor<float> A = make2d(2, 2, {1, 2, 2, 4});
    EXPECT_THROW(A.inv(), std::runtime_error);
}

TEST_F(TensorTest, Det2x2) {
    Tensor<float> A = make2d(2, 2, {1, 2, 3, 4});
    EXPECT_NEAR(A.det(), -2.f, 1e-5);
}

TEST_F(TensorTest, Det3x3) {
    Tensor<float> A = make2d(3, 3, {1,2,3, 4,5,6, 7,8,9});
    EXPECT_NEAR(A.det(), 0.f, 1e-4);
}

TEST_F(TensorTest, DetNonSquareThrows) {
    Tensor<float> A = make2d(2, 3, {1,2,3,4,5,6});
    EXPECT_THROW(A.det(), std::invalid_argument);
}

TEST_F(TensorTest, DetIdentity) {
    Tensor<float> I = Tensor<float>::eye(eng, 4);
    EXPECT_NEAR(I.det(), 1.f, 1e-5);
}

TEST_F(TensorTest, DetDiagonal) {
    Tensor<float> A = make2d(2, 2, {3, 0, 0, 7});
    EXPECT_NEAR(A.det(), 21.f, 1e-5);
}

TEST_F(TensorTest, Nrm2) {
    Tensor<float> t = make({3.f, 4.f});
    EXPECT_NEAR(t.nrm2(), 5.f, 1e-5);
}

TEST_F(TensorTest, Asum) {
    Tensor<float> t = make({-1.f, 2.f, -3.f});
    EXPECT_NEAR(t.asum(), 6.f, 1e-5);
}

TEST_F(TensorTest, Iamax) {
    Tensor<float> t = make({1.f, -5.f, 3.f});
    EXPECT_EQ(t.iamax(), 1u);
}

TEST_F(TensorTest, CopyTo) {
    Tensor<float> src = make({1, 2, 3});
    Tensor<float> dst(eng, 3);
    dst.fill(0);
    src.copy_to(dst);
    EXPECT_FLOAT_EQ(dst[0], 1.f);
    EXPECT_FLOAT_EQ(dst[2], 3.f);
}

TEST_F(TensorTest, CopyToThrows) {
    Tensor<float> src = make({1, 2});
    Tensor<float> dst(eng, 3);
    EXPECT_THROW(src.copy_to(dst), std::invalid_argument);
}

TEST_F(TensorTest, Sum) {
    EXPECT_NEAR(make({1, 2, 3}).sum(), 6.f, 1e-5);
}

TEST_F(TensorTest, Max) {
    EXPECT_FLOAT_EQ(make({3, 1, 2}).max(), 3.f);
}

TEST_F(TensorTest, Min) {
    EXPECT_FLOAT_EQ(make({3, 1, 2}).min(), 1.f);
}

TEST_F(TensorTest, Argmax) {
    EXPECT_EQ(make({3, 1, 2}).argmax(), 0u);
    EXPECT_EQ(make({1, 3, 2}).argmax(), 1u);
}

TEST_F(TensorTest, Argmin) {
    EXPECT_EQ(make({3, 1, 2}).argmin(), 1u);
}

TEST_F(TensorTest, Dot) {
    Tensor<float> a = make({1, 2, 3});
    Tensor<float> b = make({4, 5, 6});
    EXPECT_NEAR(a.dot(b), 32.f, 1e-5); // 4+10+18
}

TEST_F(TensorTest, DotThrows) {
    EXPECT_THROW(make({1, 2}).dot(make({1, 2, 3})), std::invalid_argument);
}

TEST_F(TensorTest, SumAxis0) {
    Tensor<float> t = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> r = t.sum(0);
    EXPECT_EQ(r.rows(), 1u);
    EXPECT_EQ(r.cols(), 3u);
    EXPECT_FLOAT_EQ(r[0], 5.f);
    EXPECT_FLOAT_EQ(r[1], 7.f);
    EXPECT_FLOAT_EQ(r[2], 9.f);
}

TEST_F(TensorTest, SumAxis1) {
    Tensor<float> t = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> r = t.sum(1);
    EXPECT_EQ(r.rows(), 2u);
    EXPECT_EQ(r.cols(), 1u);
    EXPECT_FLOAT_EQ(r[0], 6.f);
    EXPECT_FLOAT_EQ(r[1], 15.f);
}

TEST_F(TensorTest, MaxAxis0) {
    Tensor<float> t = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> r = t.max(0);
    EXPECT_FLOAT_EQ(r[0], 4.f);
    EXPECT_FLOAT_EQ(r[1], 5.f);
    EXPECT_FLOAT_EQ(r[2], 6.f);
}

TEST_F(TensorTest, MaxAxis1) {
    Tensor<float> t = make2d(2, 3, {1,2,3, 4,5,6});
    Tensor<float> r = t.max(1);
    EXPECT_FLOAT_EQ(r[0], 3.f);
    EXPECT_FLOAT_EQ(r[1], 6.f);
}

TEST_F(TensorTest, Exp) {
    Tensor<float> r = make({0.f, 1.f}).exp();
    EXPECT_NEAR(r[0], 1.0, 1e-5);
    EXPECT_NEAR(r[1], std::exp(1.0), 1e-5);
}

TEST_F(TensorTest, Log) {
    Tensor<float> r = make({1.f, 2.718281828f}).log();
    EXPECT_NEAR(r[0], 0.0, 1e-4);
    EXPECT_NEAR(r[1], 1.0, 1e-4);
}

TEST_F(TensorTest, Abs) {
    Tensor<float> r = make({-1.f, 0.f, 3.f}).abs();
    EXPECT_FLOAT_EQ(r[0], 1.f);
    EXPECT_FLOAT_EQ(r[1], 0.f);
    EXPECT_FLOAT_EQ(r[2], 3.f);
}

TEST_F(TensorTest, Sqrt) {
    Tensor<float> r = make({4.f, 9.f}).sqrt();
    EXPECT_NEAR(r[0], 2.0, 1e-5);
    EXPECT_NEAR(r[1], 3.0, 1e-5);
}

TEST_F(TensorTest, Neg) {
    Tensor<float> r = make({1.f, -2.f, 0.f}).neg();
    EXPECT_FLOAT_EQ(r[0], -1.f);
    EXPECT_FLOAT_EQ(r[1], 2.f);
    EXPECT_FLOAT_EQ(r[2], 0.f);
}

TEST_F(TensorTest, Fill) {
    Tensor<float> t(eng, 4);
    t.fill(7.f);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(t[i], 7.f);
}

TEST_F(TensorTest, Eye) {
    Tensor<float> I = Tensor<float>::eye(eng, 3);
    EXPECT_EQ(I.rows(), 3u);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(I(i, j), (i == j) ? 1.f : 0.f);
}

TEST_F(TensorTest, ZerosMatrix) {
    Tensor<float> z = Tensor<float>::zeros(eng, 2, 3);
    EXPECT_EQ(z.rows(), 2u);
    EXPECT_EQ(z.cols(), 3u);
    for (size_t i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(z[i], 0.f);
}

TEST_F(TensorTest, OnesMatrix) {
    Tensor<float> o = Tensor<float>::ones(eng, 3, 2);
    EXPECT_EQ(o.size(), 6u);
    for (size_t i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(o[i], 1.f);
}

TEST_F(TensorTest, ConstantMatrix) {
    Tensor<float> c = Tensor<float>::constant(eng, 2, 2, 5.f);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(c[i], 5.f);
}

TEST_F(TensorTest, ZerosVector) {
    Tensor<float> z = Tensor<float>::zeros(eng, 4);
    EXPECT_EQ(z.size(), 4u);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(z[i], 0.f);
}

TEST_F(TensorTest, OnesVector) {
    Tensor<float> o = Tensor<float>::ones(eng, 3);
    EXPECT_EQ(o.size(), 3u);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(o[i], 1.f);
}

TEST_F(TensorTest, StreamVector) {
    Tensor<float> t = make({1.f, 2.f, 3.f});
    std::ostringstream oss;
    oss << t;
    EXPECT_EQ(oss.str(), "[1, 2, 3]");
}

TEST_F(TensorTest, StreamMatrix) {
    Tensor<float> t = make2d(2, 2, {1.f, 2.f, 3.f, 4.f});
    std::ostringstream oss;
    oss << t;
    EXPECT_EQ(oss.str(), "[[1, 2]\n [3, 4]]");
}

class TensorDoubleTest : public ::testing::Test {
protected:
    CpuTensorEngine<double> eng;
};

TEST_F(TensorDoubleTest, BasicOps) {
    Tensor<double> a(eng, {1.0, 2.0, 3.0});
    Tensor<double> b(eng, {4.0, 5.0, 6.0});
    Tensor<double> r = a + b;
    EXPECT_DOUBLE_EQ(r[0], 5.0);
    EXPECT_DOUBLE_EQ(r[2], 9.0);
}

TEST_F(TensorDoubleTest, Matmul) {
    Tensor<double> A(eng, 2, 2);
    A[0] = 1; A[1] = 2; A[2] = 3; A[3] = 4;
    Tensor<double> B(eng, 2, 2);
    B[0] = 5; B[1] = 6; B[2] = 7; B[3] = 8;
    Tensor<double> C = A.matmul(B);
    EXPECT_DOUBLE_EQ(C(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(C(1, 1), 50.0);
}

TEST_F(TensorDoubleTest, InvDet) {
    Tensor<double> A(eng, 2, 2);
    A[0] = 1; A[1] = 2; A[2] = 3; A[3] = 4;
    EXPECT_NEAR(A.det(), -2.0, 1e-10);
    Tensor<double> I = A.inv();
    EXPECT_NEAR(I(0, 0), -2.0, 1e-10);
}
