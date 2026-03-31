#pragma once

#include "../tensor/tensor.h"
#include "../tensor/cpu_engine.h"
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <optional>
#include <stdexcept>
#include <limits>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <iostream>

namespace ml {

template <typename T>
class Series {
public:
    using ValueType = T;

    Series() : name_(""), engine_(std::make_shared<CpuTensorEngine<T>>()) {}
    
    explicit Series(const std::string& name) : name_(name), engine_(std::make_shared<CpuTensorEngine<T>>()) {}
    
    Series(const std::string& name, const std::vector<T>& data) 
        : name_(name), engine_(std::make_shared<CpuTensorEngine<T>>()), data_(data) {}
    
    Series(const std::string& name, std::shared_ptr<CpuTensorEngine<T>> engine)
        : name_(name), engine_(engine) {}

    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    const T& operator[](size_t i) const { return data_[i]; }
    
    template<typename U = T>
    typename std::enable_if<!std::is_same<U, bool>::value, U&>::type
    operator[](size_t i) { return data_[i]; }

    T& at(size_t i) { 
        if (i >= size()) throw std::out_of_range("Series index out of range");
        return data_[i]; 
    }
    const T& at(size_t i) const { 
        if (i >= size()) throw std::out_of_range("Series index out of range");
        return data_[i]; 
    }

    std::shared_ptr<CpuTensorEngine<T>> engine() { return engine_; }

    void push_back(const T& value) { data_.push_back(value); }
    void push_back(T&& value) { data_.push_back(std::move(value)); }
    void reserve(size_t n) { data_.reserve(n); }
    void resize(size_t n) { data_.resize(n); }
    void resize(size_t n, const T& value) { data_.resize(n, value); }

    T sum() const {
        if (data_.empty()) return T(0);
        return std::accumulate(data_.begin(), data_.end(), T(0));
    }

    T mean() const { return sum() / static_cast<T>(size()); }

    T var() const {
        if (size() < 2) return T(0);
        T m = mean();
        T sum_sq = T(0);
        for (const auto& val : data_) {
            T diff = val - m;
            sum_sq += diff * diff;
        }
        return sum_sq / static_cast<T>(size() - 1);
    }

    T std() const { return std::sqrt(var()); }
    T min() const { return *std::min_element(data_.begin(), data_.end()); }
    T max() const { return *std::max_element(data_.begin(), data_.end()); }

    T median() const {
        if (empty()) return T(0);
        std::vector<T> sorted = data_;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / T(2);
        }
        return sorted[n/2];
    }

    T quantile(T q) const {
        if (empty()) return T(0);
        if (q < 0 || q > 1) throw std::invalid_argument("Quantile must be between 0 and 1");
        std::vector<T> sorted = data_;
        std::sort(sorted.begin(), sorted.end());
        double pos = (sorted.size() - 1) * q;
        size_t idx = static_cast<size_t>(pos);
        T frac = static_cast<T>(pos - idx);
        if (idx + 1 < sorted.size()) {
            return sorted[idx] * (T(1) - frac) + sorted[idx + 1] * frac;
        }
        return sorted[idx];
    }

    T dot(const Series& other) const {
        if (size() != other.size())
            throw std::invalid_argument("Series size mismatch in dot");
        T result = T(0);
        for (size_t i = 0; i < size(); ++i) {
            result += data_[i] * other.data_[i];
        }
        return result;
    }

    size_t idxmin() const {
        return std::min_element(data_.begin(), data_.end()) - data_.begin();
    }

    size_t idxmax() const {
        return std::max_element(data_.begin(), data_.end()) - data_.begin();
    }

    Series abs() const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = std::abs(data_[i]);
        }
        return result;
    }

    Series sqrt() const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = std::sqrt(data_[i]);
        }
        return result;
    }

    Series exp() const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = std::exp(data_[i]);
        }
        return result;
    }

    Series log() const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = std::log(data_[i]);
        }
        return result;
    }

    Series neg() const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = -data_[i];
        }
        return result;
    }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void fillna(const T& value) {
        for (size_t i = 0; i < data_.size(); ++i) {
            if (std::isnan(static_cast<double>(data_[i]))) {
                data_[i] = value;
            }
        }
    }

    Series head(size_t n = 5) const {
        n = std::min(n, size());
        return Series(name_, std::vector<T>(data_.begin(), data_.begin() + n));
    }

    Series tail(size_t n = 5) const {
        n = std::min(n, size());
        return Series(name_, std::vector<T>(data_.end() - n, data_.end()));
    }

    Series diff(int periods = 1) const {
        if (size() <= static_cast<size_t>(periods)) {
            return Series(name_);
        }
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < periods; ++i) {
            result.data_[i] = T(0);
        }
        for (size_t i = periods; i < size(); ++i) {
            result.data_[i] = data_[i] - data_[i - periods];
        }
        return result;
    }

    Series cumsum() const {
        Series result(name_);
        if (empty()) return result;
        result.data_.resize(size());
        result.data_[0] = data_[0];
        for (size_t i = 1; i < size(); ++i) {
            result.data_[i] = result.data_[i-1] + data_[i];
        }
        return result;
    }

    Series cummax() const {
        Series result(name_);
        if (empty()) return result;
        result.data_.resize(size());
        result.data_[0] = data_[0];
        for (size_t i = 1; i < size(); ++i) {
            result.data_[i] = std::max(result.data_[i-1], data_[i]);
        }
        return result;
    }

    Series cummin() const {
        Series result(name_);
        if (empty()) return result;
        result.data_.resize(size());
        result.data_[0] = data_[0];
        for (size_t i = 1; i < size(); ++i) {
            result.data_[i] = std::min(result.data_[i-1], data_[i]);
        }
        return result;
    }

    Series rolling(size_t window, bool center = false) const {
        Series result(name_);
        if (size() < window) return result;
        result.data_.resize(size());
        for (size_t i = 0; i < window - 1; ++i) {
            result.data_[i] = std::numeric_limits<T>::quiet_NaN();
        }
        for (size_t i = window - 1; i < size(); ++i) {
            size_t start = center ? i - window / 2 : i - window + 1;
            T sum_val = T(0);
            for (size_t j = 0; j < window; ++j) {
                sum_val += data_[start + j];
            }
            result.data_[i] = sum_val / static_cast<T>(window);
        }
        return result;
    }

    Series ewm(double alpha = 0.3) const {
        Series result(name_);
        if (empty()) return result;
        result.data_.resize(size());
        result.data_[0] = data_[0];
        for (size_t i = 1; i < size(); ++i) {
            result.data_[i] = alpha * data_[i] + (1 - alpha) * result.data_[i-1];
        }
        return result;
    }

    Series operator+(const Series& other) const {
        if (size() != other.size()) 
            throw std::invalid_argument("Series size mismatch in +");
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Series operator-(const Series& other) const {
        if (size() != other.size()) 
            throw std::invalid_argument("Series size mismatch in -");
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }

    Series operator*(const Series& other) const {
        if (size() != other.size()) 
            throw std::invalid_argument("Series size mismatch in *");
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }

    Series operator/(const Series& other) const {
        if (size() != other.size()) 
            throw std::invalid_argument("Series size mismatch in /");
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] / other.data_[i];
        }
        return result;
    }

    Series operator+(T scalar) const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] + scalar;
        }
        return result;
    }

    Series operator-(T scalar) const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] - scalar;
        }
        return result;
    }

    Series operator*(T scalar) const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    Series operator/(T scalar) const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }

    friend Series operator+(T scalar, const Series& s) { return s + scalar; }
    friend Series operator*(T scalar, const Series& s) { return s * scalar; }

    Series& operator+=(const Series& other) {
        if (size() != other.size())
            throw std::invalid_argument("Series size mismatch in +=");
        for (size_t i = 0; i < size(); ++i) {
            data_[i] += other.data_[i];
        }
        return *this;
    }

    Series& operator-=(const Series& other) {
        if (size() != other.size())
            throw std::invalid_argument("Series size mismatch in -=");
        for (size_t i = 0; i < size(); ++i) {
            data_[i] -= other.data_[i];
        }
        return *this;
    }

    Series& operator*=(const Series& other) {
        if (size() != other.size())
            throw std::invalid_argument("Series size mismatch in *=");
        for (size_t i = 0; i < size(); ++i) {
            data_[i] *= other.data_[i];
        }
        return *this;
    }

    Series& operator/=(const Series& other) {
        if (size() != other.size())
            throw std::invalid_argument("Series size mismatch in /=");
        for (size_t i = 0; i < size(); ++i) {
            data_[i] /= other.data_[i];
        }
        return *this;
    }

    Series& operator+=(T scalar) {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] += scalar;
        }
        return *this;
    }

    Series& operator-=(T scalar) {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] -= scalar;
        }
        return *this;
    }

    Series& operator*=(T scalar) {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] *= scalar;
        }
        return *this;
    }

    Series& operator/=(T scalar) {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] /= scalar;
        }
        return *this;
    }

    Series& operator++() {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] += T(1);
        }
        return *this;
    }

    Series operator++(int) {
        Series tmp(*this);
        ++(*this);
        return tmp;
    }

    Series& operator--() {
        for (size_t i = 0; i < size(); ++i) {
            data_[i] -= T(1);
        }
        return *this;
    }

    Series operator--(int) {
        Series tmp(*this);
        --(*this);
        return tmp;
    }

    Series operator-() const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = -data_[i];
        }
        return result;
    }

    Series operator+() const { return *this; }

    Series sort(bool ascending = true) const {
        std::vector<T> sorted = data_;
        if (ascending) {
            std::sort(sorted.begin(), sorted.end());
        } else {
            std::sort(sorted.begin(), sorted.end(), std::greater<T>{});
        }
        return Series(name_, sorted);
    }

    Series rank() const {
        std::vector<size_t> indices(size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(), 
            [this](size_t a, size_t b) { return data_[a] < data_[b]; });
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[indices[i]] = static_cast<T>(i + 1);
        }
        return result;
    }

    size_t count() const {
        return data_.size();
    }

    T prod() const {
        if (empty()) return T(1);
        T result = T(1);
        for (const auto& val : data_) {
            result *= val;
        }
        return result;
    }

    Series clip(T lower = std::numeric_limits<T>::lowest(), 
                T upper = std::numeric_limits<T>::max()) const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = std::max(lower, std::min(upper, data_[i]));
        }
        return result;
    }

    std::vector<T> to_vector() const { return data_; }

    void print(std::ostream& os = std::cout) const {
        os << name_ << ":\n[";
        for (size_t i = 0; i < data_.size(); ++i) {
            if (i > 0) os << ", ";
            os << data_[i];
        }
        os << "]\n";
    }

    Series<bool> isnull() const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(std::isnan(static_cast<double>(val)));
        }
        return result;
    }

    Series<bool> notnull() const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(!std::isnan(static_cast<double>(val)));
        }
        return result;
    }

    Series filter(const Series<bool>& mask) const {
        if (size() != mask.size())
            throw std::invalid_argument("Series and mask size mismatch");
        Series result(name_);
        for (size_t i = 0; i < size(); ++i) {
            if (mask[i]) {
                result.push_back(data_[i]);
            }
        }
        return result;
    }

    Series dropna() const {
        Series result(name_);
        for (size_t i = 0; i < size(); ++i) {
            if (!std::isnan(static_cast<double>(data_[i]))) {
                result.push_back(data_[i]);
            }
        }
        return result;
    }

    template <typename F>
    Series apply(F&& func) const {
        Series result(name_);
        result.data_.resize(size());
        for (size_t i = 0; i < size(); ++i) {
            result.data_[i] = func(data_[i]);
        }
        return result;
    }

    template <typename U>
    Series<U> astype() const {
        Series<U> result(name_);
        for (const auto& val : data_) {
            result.push_back(static_cast<U>(val));
        }
        return result;
    }

    Series<bool> operator<(const Series& other) const {
        Series<bool> result(name_);
        for (size_t i = 0; i < size(); ++i) {
            result.push_back(data_[i] < other.data_[i]);
        }
        return result;
    }

    Series<bool> operator>(const Series& other) const {
        Series<bool> result(name_);
        for (size_t i = 0; i < size(); ++i) {
            result.push_back(data_[i] > other.data_[i]);
        }
        return result;
    }

    Series<bool> operator<=(const Series& other) const {
        Series<bool> result(name_);
        for (size_t i = 0; i < size(); ++i) {
            result.push_back(data_[i] <= other.data_[i]);
        }
        return result;
    }

    Series<bool> operator>=(const Series& other) const {
        Series<bool> result(name_);
        for (size_t i = 0; i < size(); ++i) {
            result.push_back(data_[i] >= other.data_[i]);
        }
        return result;
    }

    Series<bool> operator==(const Series& other) const {
        Series<bool> result(name_);
        for (size_t i = 0; i < size(); ++i) {
            result.push_back(data_[i] == other.data_[i]);
        }
        return result;
    }

    Series<bool> operator!=(const Series& other) const {
        Series<bool> result(name_);
        for (size_t i = 0; i < size(); ++i) {
            result.push_back(data_[i] != other.data_[i]);
        }
        return result;
    }

    Series<bool> operator<(T scalar) const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(val < scalar);
        }
        return result;
    }

    Series<bool> operator>(T scalar) const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(val > scalar);
        }
        return result;
    }

    Series<bool> operator<=(T scalar) const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(val <= scalar);
        }
        return result;
    }

    Series<bool> operator>=(T scalar) const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(val >= scalar);
        }
        return result;
    }

    Series<bool> operator==(T scalar) const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(val == scalar);
        }
        return result;
    }

    Series<bool> operator!=(T scalar) const {
        Series<bool> result(name_);
        for (const auto& val : data_) {
            result.push_back(val != scalar);
        }
        return result;
    }

private:
    std::string name_;
    std::vector<T> data_;
    std::shared_ptr<CpuTensorEngine<T>> engine_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Series<T>& s) {
    s.print(os);
    return os;
}

using SeriesDouble = Series<double>;
using SeriesFloat = Series<float>;
using SeriesInt = Series<int>;
using SeriesBool = Series<bool>;

} // namespace ml
