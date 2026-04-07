#pragma once

#include "frame.h"
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>

namespace ml {

namespace stats {

template <typename T>
T mean(const std::vector<T>& data) {
    if (data.empty()) return T(0);
    return std::accumulate(data.begin(), data.end(), T(0)) / static_cast<T>(data.size());
}

template <typename T>
T variance(const std::vector<T>& data) {
    if (data.size() < 2) return T(0);
    T m = mean(data);
    T sum_sq = T(0);
    for (const auto& val : data) {
        T diff = val - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<T>(data.size() - 1);
}

template <typename T>
T std_dev(const std::vector<T>& data) {
    return std::sqrt(variance(data));
}

template <typename T>
T median(std::vector<T> data) {
    if (data.empty()) return T(0);
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0) {
        return (data[n/2 - 1] + data[n/2]) / T(2);
    }
    return data[n/2];
}

template <typename T>
T percentile(std::vector<T> data, T p) {
    if (data.empty()) return T(0);
    if (p < 0 || p > 1) throw std::invalid_argument("Percentile must be between 0 and 1");
    std::sort(data.begin(), data.end());
    double pos = (data.size() - 1) * p;
    size_t idx = static_cast<size_t>(pos);
    T frac = static_cast<T>(pos - idx);
    if (idx + 1 < data.size()) {
        return data[idx] * (T(1) - frac) + data[idx + 1] * frac;
    }
    return data[idx];
}

template <typename T>
T covariance(const std::vector<T>& x, const std::vector<T>& y) {
    if (x.size() != y.size() || x.size() < 2) return T(0);
    T mean_x = mean(x);
    T mean_y = mean(y);
    T sum = T(0);
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return sum / static_cast<T>(x.size() - 1);
}

template <typename T>
T correlation(const std::vector<T>& x, const std::vector<T>& y) {
    T cov = covariance(x, y);
    T std_x = std_dev(x);
    T std_y = std_dev(y);
    if (std_x == 0 || std_y == 0) return T(0);
    return cov / (std_x * std_y);
}

template <typename T>
T skewness(const std::vector<T>& data) {
    if (data.size() < 3) return T(0);
    T m = mean(data);
    T s = std_dev(data);
    if (s == 0) return T(0);
    T sum_cubed = T(0);
    for (const auto& val : data) {
        T diff = (val - m) / s;
        sum_cubed += diff * diff * diff;
    }
    T n = static_cast<T>(data.size());
    return (n / ((n - 1) * (n - 2))) * sum_cubed;
}

template <typename T>
T kurtosis(const std::vector<T>& data) {
    if (data.size() < 4) return T(0);
    T m = mean(data);
    T s = std_dev(data);
    if (s == 0) return T(0);
    T sum_quartile = T(0);
    for (const auto& val : data) {
        T diff = (val - m) / s;
        sum_quartile += diff * diff * diff * diff;
    }
    T n = static_cast<T>(data.size());
    T coef = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3));
    T subtract = 3 * (n - 1) * (n - 1) / ((n - 2) * (n - 3));
    return coef * sum_quartile - subtract;
}

template <typename T>
std::map<std::string, T> describe(const std::vector<T>& data) {
    std::map<std::string, T> result;
    result["count"] = static_cast<T>(data.size());
    result["mean"] = mean(data);
    result["std"] = std_dev(data);
    result["min"] = *std::min_element(data.begin(), data.end());
    result["max"] = *std::max_element(data.begin(), data.end());
    result["25%"] = percentile(data, T(0.25));
    result["50%"] = percentile(data, T(0.50));
    result["75%"] = percentile(data, T(0.75));
    return result;
}

template <typename T>
std::vector<T> normalize(const std::vector<T>& data) {
    if (data.empty()) return {};
    T min_val = *std::min_element(data.begin(), data.end());
    T max_val = *std::max_element(data.begin(), data.end());
    if (max_val == min_val) return std::vector<T>(data.size(), T(0));
    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = (data[i] - min_val) / (max_val - min_val);
    }
    return result;
}

template <typename T>
std::vector<T> standardize(const std::vector<T>& data) {
    T m = mean(data);
    T s = std_dev(data);
    if (s == 0) return std::vector<T>(data.size(), T(0));
    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = (data[i] - m) / s;
    }
    return result;
}

template <typename T>
std::vector<T> log_transform(const std::vector<T>& data) {
    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = std::log(data[i] + T(1));
    }
    return result;
}

template <typename T>
std::vector<T> sqrt_transform(const std::vector<T>& data) {
    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = std::sqrt(std::abs(data[i]));
    }
    return result;
}

template <typename T>
std::vector<T> box_cox_transform(const std::vector<T>& data, T lambda) {
    std::vector<T> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (lambda == 0) {
            result[i] = std::log(data[i]);
        } else {
            result[i] = (std::pow(data[i], lambda) - 1) / lambda;
        }
    }
    return result;
}

template <typename T>
T z_score(const T& value, const std::vector<T>& data) {
    T m = mean(data);
    T s = std_dev(data);
    if (s == 0) return T(0);
    return (value - m) / s;
}

template <typename T>
std::vector<size_t> argsort(const std::vector<T>& data, bool ascending = true) {
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (ascending) {
        std::stable_sort(indices.begin(), indices.end(), 
            [&data](size_t a, size_t b) { return data[a] < data[b]; });
    } else {
        std::stable_sort(indices.begin(), indices.end(), 
            [&data](size_t a, size_t b) { return data[a] > data[b]; });
    }
    return indices;
}

template <typename T>
std::vector<T> bootstrap(const std::vector<T>& data, size_t n_iterations, unsigned int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    std::vector<T> means;
    for (size_t i = 0; i < n_iterations; ++i) {
        std::vector<T> sample;
        for (size_t j = 0; j < data.size(); ++j) {
            sample.push_back(data[dist(gen)]);
        }
        means.push_back(mean(sample));
    }
    return means;
}

template <typename T>
T confidence_interval(const std::vector<T>& data, T confidence = 0.95) {
    T m = mean(data);
    T se = std_dev(data) / std::sqrt(static_cast<T>(data.size()));
    T critical = percentile(bootstrap(data, 1000), T(1 - (1 - confidence) / 2));
    return critical * se;
}

template <typename T>
std::vector<std::vector<T>> covariance_matrix(const std::vector<std::vector<T>>& data) {
    size_t n = data.size();
    std::vector<std::vector<T>> result(n, std::vector<T>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i][j] = covariance(data[i], data[j]);
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> correlation_matrix(const std::vector<std::vector<T>>& data) {
    size_t n = data.size();
    std::vector<std::vector<T>> result(n, std::vector<T>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i][j] = correlation(data[i], data[j]);
        }
    }
    return result;
}

template <typename T>
DataFrame<T> describe_dataframe(const DataFrame<T>& df) {
    DataFrame<T> result;
    for (size_t i = 0; i < df.cols(); ++i) {
        const auto& col = df.col(i);
        auto stats = describe(col.to_vector());
        std::vector<std::string> stat_names;
        std::vector<T> stat_values;
        for (const auto& [key, val] : stats) {
            stat_names.push_back(key);
            stat_values.push_back(val);
        }
        result.add_column("column_" + std::to_string(i), stat_values);
    }
    return result;
}

} // namespace stats

} // namespace ml
