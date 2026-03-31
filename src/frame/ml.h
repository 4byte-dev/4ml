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
namespace ml_ops {

template <typename T>
struct TrainTestSplit {
    DataFrame<T> X_train;
    DataFrame<T> X_test;
    DataFrame<T> y_train;
    DataFrame<T> y_test;
};

template <typename T>
TrainTestSplit<T> train_test_split(const DataFrame<T>& X, const DataFrame<T>& y, 
                                   double test_size = 0.2, unsigned int seed = 0) {
    std::mt19937 gen(seed);
    size_t n_samples = X.rows();
    size_t n_test = static_cast<size_t>(n_samples * test_size);
    size_t n_train = n_samples - n_test;
    
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    std::vector<size_t> train_indices(indices.begin(), indices.begin() + n_train);
    std::vector<size_t> test_indices(indices.begin() + n_train, indices.end());
    
    TrainTestSplit<T> result;
    result.X_train = X.iloc(train_indices);
    result.X_test = X.iloc(test_indices);
    result.y_train = y.iloc(train_indices);
    result.y_test = y.iloc(test_indices);
    
    return result;
}

template <typename T>
DataFrame<T> min_max_scaler(const DataFrame<T>& data) {
    DataFrame<T> result = data;
    for (size_t i = 0; i < result.cols(); ++i) {
        auto& col = result.col(i);
        T min_val = col.min();
        T max_val = col.max();
        if (max_val != min_val) {
            for (size_t j = 0; j < col.size(); ++j) {
                col[j] = (col[j] - min_val) / (max_val - min_val);
            }
        }
    }
    return result;
}

template <typename T>
DataFrame<T> standard_scaler(const DataFrame<T>& data) {
    DataFrame<T> result = data;
    for (size_t i = 0; i < result.cols(); ++i) {
        auto& col = result.col(i);
        T mean_val = col.mean();
        T std_val = col.std();
        if (std_val != 0) {
            for (size_t j = 0; j < col.size(); ++j) {
                col[j] = (col[j] - mean_val) / std_val;
            }
        }
    }
    return result;
}

template <typename T>
std::vector<T> one_hot_encode(const std::vector<T>& labels) {
    std::vector<T> unique_labels = labels;
    std::sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
    
    std::map<T, size_t> label_to_idx;
    for (size_t i = 0; i < unique_labels.size(); ++i) {
        label_to_idx[unique_labels[i]] = i;
    }
    
    std::vector<T> encoded(labels.size() * unique_labels.size(), T(0));
    for (size_t i = 0; i < labels.size(); ++i) {
        size_t idx = label_to_idx[labels[i]];
        encoded[i * unique_labels.size() + idx] = T(1);
    }
    
    return encoded;
}

template <typename T>
std::vector<size_t> label_encode(const std::vector<T>& labels) {
    std::vector<T> unique_labels = labels;
    std::sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
    
    std::map<T, size_t> label_to_idx;
    for (size_t i = 0; i < unique_labels.size(); ++i) {
        label_to_idx[unique_labels[i]] = i;
    }
    
    std::vector<size_t> encoded(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        encoded[i] = label_to_idx[labels[i]];
    }
    
    return encoded;
}

template <typename T>
DataFrame<T> polynomial_features(const DataFrame<T>& data, int degree = 2) {
    DataFrame<T> result = data;
    
    for (size_t i = 0; i < data.cols(); ++i) {
        for (int d = 2; d <= degree; ++d) {
            std::vector<T> new_col;
            const auto& col = data.col(i);
            for (size_t j = 0; j < col.size(); ++j) {
                new_col.push_back(std::pow(col[j], d));
            }
            result.add_column(data.columns(i) + "_pow" + std::to_string(d), new_col);
        }
    }
    
    return result;
}

template <typename T>
DataFrame<T> cross_terms(const DataFrame<T>& data) {
    DataFrame<T> result = data;
    
    for (size_t i = 0; i < data.cols(); ++i) {
        for (size_t j = i + 1; j < data.cols(); ++j) {
            std::vector<T> new_col;
            const auto& col1 = data.col(i);
            const auto& col2 = data.col(j);
            for (size_t k = 0; k < col1.size(); ++k) {
                new_col.push_back(col1[k] * col2[k]);
            }
            result.add_column(data.columns(i) + "_x_" + data.columns(j), new_col);
        }
    }
    
    return result;
}

template <typename T>
std::vector<T> k_means_clustering(const std::vector<T>& data, size_t k, size_t max_iter = 100, unsigned int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    
    std::vector<T> centroids(k);
    for (size_t i = 0; i < k; ++i) {
        centroids[i] = data[dist(gen)];
    }
    
    std::vector<size_t> assignments(data.size());
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
        for (size_t i = 0; i < data.size(); ++i) {
            T min_dist = std::abs(data[i] - centroids[0]);
            size_t min_idx = 0;
            for (size_t j = 1; j < k; ++j) {
                T dist = std::abs(data[i] - centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            assignments[i] = min_idx;
        }
        
        std::vector<T> new_centroids(k, T(0));
        std::vector<size_t> counts(k, 0);
        
        for (size_t i = 0; i < data.size(); ++i) {
            new_centroids[assignments[i]] += data[i];
            counts[assignments[i]]++;
        }
        
        for (size_t i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                new_centroids[i] /= static_cast<T>(counts[i]);
            }
        }
        
        centroids = new_centroids;
    }
    
    return centroids;
}

template <typename T>
T linear_regression_predict(const std::vector<T>& X, const std::vector<T>& weights, T bias) {
    T prediction = bias;
    for (size_t i = 0; i < X.size(); ++i) {
        prediction += X[i] * weights[i];
    }
    return prediction;
}

template <typename T>
std::pair<std::vector<T>, T> linear_regression_fit(const DataFrame<T>& X, const Series<T>& y, 
                                                    double learning_rate = 0.01, 
                                                    size_t n_iterations = 1000) {
    size_t n_features = X.cols();
    size_t n_samples = X.rows();
    
    std::vector<T> weights(n_features, T(0));
    T bias = T(0);
    
    for (size_t iter = 0; iter < n_iterations; ++iter) {
        std::vector<T> dw(n_features, T(0));
        T db = T(0);
        
        for (size_t i = 0; i < n_samples; ++i) {
            T prediction = bias;
            for (size_t j = 0; j < n_features; ++j) {
                prediction += X.at(i, j) * weights[j];
            }
            T error = prediction - y[i];
            
            db += error / static_cast<T>(n_samples);
            for (size_t j = 0; j < n_features; ++j) {
                dw[j] += error * X.at(i, j) / static_cast<T>(n_samples);
            }
        }
        
        bias -= learning_rate * db;
        for (size_t j = 0; j < n_features; ++j) {
            weights[j] -= learning_rate * dw[j];
        }
    }
    
    return {weights, bias};
}

template <typename T>
T linear_regression_predict_row(const DataFrame<T>& X, size_t row_idx, 
                                const std::vector<T>& weights, T bias);

template <typename T>
DataFrame<T> linear_regression_predict(const DataFrame<T>& X, const std::vector<T>& weights, T bias) {
    DataFrame<T> predictions;
    std::vector<T> pred_values;
    
    for (size_t i = 0; i < X.rows(); ++i) {
        T pred = linear_regression_predict_row(X, i, weights, bias);
        pred_values.push_back(pred);
    }
    
    predictions.add_column("prediction", pred_values);
    return predictions;
}

template <typename T>
T linear_regression_predict_row(const DataFrame<T>& X, size_t row_idx, 
                                const std::vector<T>& weights, T bias) {
    T prediction = bias;
    for (size_t j = 0; j < X.cols(); ++j) {
        prediction += X.at(row_idx, j) * weights[j];
    }
    return prediction;
}

template <typename T>
T logistic_regression_predict(const std::vector<T>& X, const std::vector<T>& weights, T bias) {
    T z = bias;
    for (size_t i = 0; i < X.size(); ++i) {
        z += X[i] * weights[i];
    }
    return T(1) / (T(1) + std::exp(-z));
}

template <typename T>
DataFrame<T> train_test_split_X_y(const DataFrame<T>& X, const Series<T>& y, 
                                   double test_size = 0.2, unsigned int seed = 0) {
    auto split = train_test_split(X, DataFrame<T>(), test_size, seed);
    DataFrame<T> result;
    result = split.X_train;
    return result;
}

template <typename T>
DataFrame<T> k_fold_split(const DataFrame<T>& data, size_t k, size_t fold) {
    if (fold >= k) throw std::invalid_argument("Fold index out of range");
    size_t n_samples = data.rows();
    size_t fold_size = n_samples / k;
    
    size_t start = fold * fold_size;
    size_t end = (fold == k - 1) ? n_samples : start + fold_size;
    
    std::vector<size_t> test_indices(end - start);
    std::iota(test_indices.begin(), test_indices.end(), start);
    
    std::vector<size_t> train_indices;
    for (size_t i = 0; i < n_samples; ++i) {
        if (i < start || i >= end) {
            train_indices.push_back(i);
        }
    }
    
    DataFrame<T> train_result = data.iloc(train_indices);
    return train_result;
}

template <typename T>
T mean_squared_error(const Series<T>& y_true, const Series<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    T sum_sq = T(0);
    for (size_t i = 0; i < y_true.size(); ++i) {
        T diff = y_true[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<T>(y_true.size());
}

template <typename T>
T root_mean_squared_error(const Series<T>& y_true, const Series<T>& y_pred) {
    return std::sqrt(mean_squared_error(y_true, y_pred));
}

template <typename T>
T mean_absolute_error(const Series<T>& y_true, const Series<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    T sum_abs = T(0);
    for (size_t i = 0; i < y_true.size(); ++i) {
        sum_abs += std::abs(y_true[i] - y_pred[i]);
    }
    return sum_abs / static_cast<T>(y_true.size());
}

template <typename T>
T r2_score(const Series<T>& y_true, const Series<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    T mean_true = y_true.mean();
    T ss_tot = T(0);
    T ss_res = T(0);
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_tot += (y_true[i] - mean_true) * (y_true[i] - mean_true);
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }
    return T(1) - (ss_res / ss_tot);
}

template <typename T>
T accuracy_score(const Series<T>& y_true, const Series<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Size mismatch");
    }
    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return static_cast<T>(correct) / static_cast<T>(y_true.size());
}

template <typename T>
std::map<std::string, T> classification_report(const Series<T>& y_true, const Series<T>& y_pred) {
    std::map<std::string, T> report;
    report["accuracy"] = accuracy_score(y_true, y_pred);
    report["precision"] = accuracy_score(y_true, y_pred);
    report["recall"] = accuracy_score(y_true, y_pred);
    report["f1"] = accuracy_score(y_true, y_pred);
    return report;
}

template <typename T>
DataFrame<T> decision_tree_predict(const DataFrame<T>& X, 
                                   const std::map<std::string, T>& tree,
                                   const std::string& target_column) {
    DataFrame<T> predictions;
    std::vector<T> pred_values;
    
    for (size_t i = 0; i < X.rows(); ++i) {
        T prediction = tree.at(target_column);
        pred_values.push_back(prediction);
    }
    
    predictions.add_column("prediction", pred_values);
    return predictions;
}

template <typename T>
DataFrame<T> random_forest_predict(const DataFrame<T>& X,
                                    const std::vector<std::map<std::string, T>>& trees,
                                    const std::string& target_column) {
    DataFrame<T> predictions;
    std::vector<T> pred_values;
    
    for (size_t i = 0; i < X.rows(); ++i) {
        T sum = T(0);
        for (const auto& tree : trees) {
            sum += tree.at(target_column);
        }
        pred_values.push_back(sum / static_cast<T>(trees.size()));
    }
    
    predictions.add_column("prediction", pred_values);
    return predictions;
}

template <typename T>
DataFrame<T> gradient_boosting_predict(const DataFrame<T>& X,
                                      const std::vector<std::pair<T, std::vector<T>>>& models,
                                      T initial_prediction) {
    DataFrame<T> predictions;
    std::vector<T> pred_values(X.rows(), initial_prediction);
    
    for (const auto& [learning_rate, weights] : models) {
        for (size_t i = 0; i < X.rows(); ++i) {
            T pred = T(0);
            for (size_t j = 0; j < weights.size(); ++j) {
                pred += X.at(i, j) * weights[j];
            }
            pred_values[i] += learning_rate * pred;
        }
    }
    
    predictions.add_column("prediction", pred_values);
    return predictions;
}

template <typename T>
DataFrame<T> normalize_inplace(DataFrame<T>& data) {
    return min_max_scaler(data);
}

template <typename T>
DataFrame<T> standardize_inplace(DataFrame<T>& data) {
    return standard_scaler(data);
}

template <typename T>
DataFrame<T> add_bias_term(const DataFrame<T>& data) {
    DataFrame<T> result = data;
    std::vector<T> ones(data.rows(), T(1));
    result.add_column("bias", ones);
    return result;
}

} // namespace ml_ops

} // namespace ml
