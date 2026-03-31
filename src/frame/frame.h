#pragma once

#include "../tensor/tensor.h"
#include "../tensor/cpu_engine.h"
#include "series.h"
#include "io/csv_parser.h"
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <future>
#include <random>
#include <cmath>
#include <set>

namespace ml {

template <typename T>
class DataFrame {
public:
    using ColumnType = Series<T>;

    DataFrame() : rows_(0), engine_(std::make_shared<CpuTensorEngine<T>>()) {}
    
    explicit DataFrame(size_t rows, size_t cols) : rows_(rows), engine_(std::make_shared<CpuTensorEngine<T>>()) {
        for (size_t i = 0; i < cols; ++i) {
            column_names_.push_back("column_" + std::to_string(i));
            columns_.emplace_back("column_" + std::to_string(i), engine_);
        }
        rebuild_index();
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return columns_.size(); }
    bool empty() const { return rows_ == 0; }

    const std::vector<std::string>& columns() const { return column_names_; }
    const std::string& columns(size_t i) const { return column_names_[i]; }

    Series<T>& add_column(const std::string& name) {
        if (column_index_.count(name)) {
            throw std::invalid_argument("Column already exists: " + name);
        }
        column_names_.push_back(name);
        columns_.emplace_back(name, engine_);
        column_index_[name] = columns_.size() - 1;
        return columns_.back();
    }

    Series<T>& add_column(const std::string& name, const std::vector<T>& data) {
        auto& col = add_column(name);
        for (const auto& val : data) {
            col.push_back(val);
        }
        if (rows_ == 0) {
            rows_ = col.size();
        } else if (col.size() != rows_) {
            throw std::invalid_argument("Row count mismatch");
        }
        return col;
    }

    void remove_column(const std::string& name) {
        auto it = column_index_.find(name);
        if (it == column_index_.end()) {
            throw std::invalid_argument("Column not found: " + name);
        }
        column_names_.erase(column_names_.begin() + it->second);
        columns_.erase(columns_.begin() + it->second);
        rebuild_index();
    }

    Series<T>& col(const std::string& name) {
        auto it = column_index_.find(name);
        if (it == column_index_.end()) {
            throw std::invalid_argument("Column not found: " + name);
        }
        return columns_[it->second];
    }

    const Series<T>& col(const std::string& name) const {
        auto it = column_index_.find(name);
        if (it == column_index_.end()) {
            throw std::invalid_argument("Column not found: " + name);
        }
        return columns_[it->second];
    }

    Series<T>& col(size_t idx) {
        if (idx >= columns_.size()) {
            throw std::out_of_range("Column index out of range");
        }
        return columns_[idx];
    }

    const Series<T>& col(size_t idx) const {
        if (idx >= columns_.size()) {
            throw std::out_of_range("Column index out of range");
        }
        return columns_[idx];
    }

    DataFrame select(const std::vector<std::string>& cols) const {
        DataFrame result;
        for (const auto& name : cols) {
            auto it = column_index_.find(name);
            if (it == column_index_.end()) {
                throw std::invalid_argument("Column not found: " + name);
            }
            result.columns_.push_back(columns_[it->second]);
            result.column_names_.push_back(name);
        }
        result.rows_ = rows_;
        result.rebuild_index();
        return result;
    }

    DataFrame head(size_t n) const {
        n = std::min(n, rows_);
        return iloc(0, n);
    }

    DataFrame tail(size_t n) const {
        n = std::min(n, rows_);
        return iloc(rows_ - n, rows_);
    }

    DataFrame iloc(size_t row_idx) const {
        return iloc(std::vector<size_t>{row_idx});
    }

    DataFrame iloc(size_t start_row, size_t end_row) const {
        std::vector<size_t> row_idxs(end_row - start_row);
        std::iota(row_idxs.begin(), row_idxs.end(), start_row);
        return iloc(row_idxs);
    }

    DataFrame iloc(const std::vector<size_t>& row_idxs) const {
        DataFrame result;
        result.column_names_ = column_names_;
        result.columns_ = columns_;
        result.rows_ = row_idxs.size();
        result.engine_ = engine_;
        result.rebuild_index();

        for (size_t i = 0; i < result.columns_.size(); ++i) {
            Series<T> sampled(result.column_names_[i], result.engine_);
            for (size_t idx : row_idxs) {
                if (idx < columns_[i].size()) {
                    sampled.push_back(columns_[i][idx]);
                }
            }
            result.columns_[i] = std::move(sampled);
        }
        return result;
    }

    DataFrame sort_values(const std::string& by, bool ascending = true) const {
        auto it = column_index_.find(by);
        if (it == column_index_.end()) {
            throw std::invalid_argument("Column not found: " + by);
        }

        std::vector<size_t> indices(rows_);
        std::iota(indices.begin(), indices.end(), 0);

        const auto& key_col = columns_[it->second];
        std::vector<T> key_data = key_col.to_vector();
        
        std::sort(indices.begin(), indices.end(), 
            [ascending, &key_data](size_t a, size_t b) {
                return ascending ? (key_data[a] < key_data[b]) : (key_data[a] > key_data[b]);
            });

        return iloc(indices);
    }

    DataFrame dropna() const {
        std::vector<size_t> keep_idxs;
        for (size_t i = 0; i < rows_; ++i) {
            bool keep = true;
            for (const auto& col : columns_) {
                if (i < col.size() && std::isnan(static_cast<double>(col[i]))) {
                    keep = false;
                    break;
                }
            }
            if (keep) keep_idxs.push_back(i);
        }
        return iloc(keep_idxs);
    }

    DataFrame fillna(T value) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col.fillna(value);
        }
        return result;
    }

    std::optional<size_t> index_of(const std::string& column) const {
        auto it = column_index_.find(column);
        if (it != column_index_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    T at(size_t row, size_t col) const {
        if (col >= columns_.size()) {
            throw std::out_of_range("Column index out of range");
        }
        if (row >= columns_[col].size()) {
            throw std::out_of_range("Row index out of range");
        }
        return columns_[col][row];
    }

    bool contains(const std::string& column) const {
        return column_index_.count(column) > 0;
    }

    void to_csv(const std::string& filename, char delimiter = ',') const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        to_csv(file, delimiter);
    }

    void to_csv(std::ostream& os, char delimiter = ',') const {
        for (size_t i = 0; i < column_names_.size(); ++i) {
            if (i > 0) os << delimiter;
            os << column_names_[i];
        }
        os << "\n";

        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < columns_.size(); ++c) {
                if (c > 0) os << delimiter;
                if (r < columns_[c].size()) {
                    os << columns_[c][r];
                }
            }
            os << "\n";
        }
    }

    static DataFrame from_csv(const std::string& filename, char delimiter = ',',
                              bool header = true, size_t skip_rows = 0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        return from_csv(file, delimiter, header, skip_rows);
    }

    static DataFrame from_csv(std::istream& is, char delimiter = ',',
                             bool header = true, size_t skip_rows = 0) {
        DataFrame result;
        auto engine = std::make_shared<CpuTensorEngine<T>>();
        
        std::string line;
        for (size_t i = 0; i < skip_rows && std::getline(is, line); ++i) {}
        
        if (header && std::getline(is, line)) {
            std::vector<std::string> headers = CSVParser::parse_line(line, delimiter);
            for (const auto& h : headers) {
                result.column_names_.push_back(h);
            }
        }
        
        std::vector<std::vector<T>> data_cols(result.column_names_.size());
        
        while (std::getline(is, line)) {
            auto values = CSVParser::parse_line(line, delimiter);
            for (size_t i = 0; i < std::min(values.size(), result.column_names_.size()); ++i) {
                try {
                    data_cols[i].push_back(static_cast<T>(std::stod(values[i])));
                } catch (...) {
                    data_cols[i].push_back(std::nan(""));
                }
            }
        }
        
        result.engine_ = engine;
        for (size_t i = 0; i < result.column_names_.size(); ++i) {
            Series<T> col(result.column_names_[i], engine);
            for (const auto& val : data_cols[i]) {
                col.push_back(val);
            }
            result.columns_.push_back(col);
        }
        
        if (!result.column_names_.empty()) {
            result.rows_ = data_cols[0].size();
        }
        result.rebuild_index();
        result.index_.resize(result.rows_);
        for (size_t i = 0; i < result.rows_; ++i) {
            result.index_[i] = std::to_string(i);
        }
        
        return result;
    }

    void print(std::ostream& os = std::cout, size_t max_rows = 10) const {
        os << to_string(max_rows);
    }

    std::string to_string(size_t max_rows = 10) const {
        std::ostringstream oss;
        
        size_t display_rows = std::min(rows_, max_rows);
        
        oss << std::string(15 * columns_.size() + 2, '-') << "\n";
        for (size_t j = 0; j < column_names_.size(); ++j) {
            oss << std::setw(14) << std::left << column_names_[j] << " |";
        }
        oss << "\n" << std::string(15 * columns_.size() + 2, '-') << "\n";
        
        for (size_t i = 0; i < display_rows; ++i) {
            for (size_t j = 0; j < columns_.size(); ++j) {
                if (i < columns_[j].size()) {
                    oss << std::setw(14) << std::left << columns_[j][i] << " |";
                } else {
                    oss << std::setw(14) << std::left << "NaN" << " |";
                }
            }
            oss << "\n";
        }
        
        if (rows_ > display_rows) {
            oss << "... " << (rows_ - display_rows) << " more rows\n";
        }
        oss << std::string(15 * columns_.size() + 2, '-') << "\n";
        
        return oss.str();
    }

    DataFrame operator+(const DataFrame& other) const {
        DataFrame result = *this;
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            result.columns_[i] = columns_[i] + other.columns_[i];
        }
        return result;
    }

    DataFrame operator-(const DataFrame& other) const {
        DataFrame result = *this;
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            result.columns_[i] = columns_[i] - other.columns_[i];
        }
        return result;
    }

    DataFrame operator*(const DataFrame& other) const {
        DataFrame result = *this;
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            result.columns_[i] = columns_[i] * other.columns_[i];
        }
        return result;
    }

    DataFrame operator/(const DataFrame& other) const {
        DataFrame result = *this;
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            result.columns_[i] = columns_[i] / other.columns_[i];
        }
        return result;
    }

    DataFrame operator+(T scalar) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col + scalar;
        }
        return result;
    }

    DataFrame operator-(T scalar) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col - scalar;
        }
        return result;
    }

    DataFrame operator*(T scalar) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col * scalar;
        }
        return result;
    }

    DataFrame operator/(T scalar) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col / scalar;
        }
        return result;
    }

    DataFrame& operator+=(const DataFrame& other) {
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            columns_[i] += other.columns_[i];
        }
        return *this;
    }

    DataFrame& operator-=(const DataFrame& other) {
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            columns_[i] -= other.columns_[i];
        }
        return *this;
    }

    DataFrame& operator*=(const DataFrame& other) {
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            columns_[i] *= other.columns_[i];
        }
        return *this;
    }

    DataFrame& operator/=(const DataFrame& other) {
        for (size_t i = 0; i < std::min(columns_.size(), other.columns_.size()); ++i) {
            columns_[i] /= other.columns_[i];
        }
        return *this;
    }

    DataFrame& operator+=(T scalar) {
        for (auto& col : columns_) {
            col += scalar;
        }
        return *this;
    }

    DataFrame& operator-=(T scalar) {
        for (auto& col : columns_) {
            col -= scalar;
        }
        return *this;
    }

    DataFrame& operator*=(T scalar) {
        for (auto& col : columns_) {
            col *= scalar;
        }
        return *this;
    }

    DataFrame& operator/=(T scalar) {
        for (auto& col : columns_) {
            col /= scalar;
        }
        return *this;
    }

    DataFrame cumsum() const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col.cumsum();
        }
        return result;
    }

    DataFrame abs() const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col.abs();
        }
        return result;
    }

    T cov(const std::string& col1, const std::string& col2) const {
        auto it1 = column_index_.find(col1);
        auto it2 = column_index_.find(col2);
        if (it1 == column_index_.end() || it2 == column_index_.end()) {
            throw std::invalid_argument("Column not found");
        }
        
        const auto& series1 = columns_[it1->second];
        const auto& series2 = columns_[it2->second];
        
        double mean1 = series1.mean();
        double mean2 = series2.mean();
        double sum = 0;
        for (size_t i = 0; i < std::min(series1.size(), series2.size()); ++i) {
            sum += (series1[i] - mean1) * (series2[i] - mean2);
        }
        return sum / (std::min(series1.size(), series2.size()) - 1);
    }

    T corr(const std::string& col1, const std::string& col2) const {
        T c = this->cov(col1, col2);
        double std1 = this->col(col1).std();
        double std2 = this->col(col2).std();
        if (std1 > 0 && std2 > 0) {
            return c / (std1 * std2);
        }
        return 0;
    }

    DataFrame describe() const {
        DataFrame result;
        result.engine_ = engine_;
        for (const auto& col : columns_) {
            Series<T> stats_col(col.name(), engine_);
            stats_col.push_back(col.mean());
            stats_col.push_back(col.std());
            stats_col.push_back(col.min());
            stats_col.push_back(col.max());
            result.columns_.push_back(stats_col);
            result.column_names_.push_back(col.name());
        }
        result.rows_ = 1;
        result.rebuild_index();
        return result;
    }

    DataFrame agg(const std::map<std::string, std::string>& ops) const {
        DataFrame result;
        result.engine_ = engine_;
        for (const auto& [col_name, op] : ops) {
            auto it = column_index_.find(col_name);
            if (it != column_index_.end()) {
                const auto& col = columns_[it->second];
                Series<T> result_col(col_name, engine_);
                if (op == "mean") {
                    result_col.push_back(col.mean());
                } else if (op == "sum") {
                    result_col.push_back(col.sum());
                } else if (op == "min") {
                    result_col.push_back(col.min());
                } else if (op == "max") {
                    result_col.push_back(col.max());
                } else if (op == "std") {
                    result_col.push_back(col.std());
                }
                result.columns_.push_back(result_col);
                result.column_names_.push_back(col_name);
            }
        }
        result.rows_ = 1;
        result.rebuild_index();
        return result;
    }

    DataFrame concat(const DataFrame& other) const {
        DataFrame result = *this;
        for (size_t i = 0; i < other.columns_.size() && i < result.columns_.size(); ++i) {
            for (size_t j = 0; j < other.columns_[i].size(); ++j) {
                result.columns_[i].push_back(other.columns_[i][j]);
            }
        }
        result.rows_ += other.rows_;
        return result;
    }

    void info() const {
        std::cout << "DataFrame: " << rows_ << " rows x " << cols() << " columns\n";
        std::cout << "Columns:\n";
        for (size_t i = 0; i < column_names_.size(); ++i) {
            std::cout << "  " << column_names_[i] << " (float64)\n";
        }
        std::cout << "Memory usage: " << memory_usage() << " bytes\n";
    }

    size_t memory_usage() const {
        size_t total = sizeof(*this);
        for (const auto& col : columns_) {
            total += col.size() * sizeof(T);
        }
        return total;
    }

    const std::vector<std::string>& index() const { return index_; }

    void set_index(const std::vector<std::string>& idx) { index_ = idx; }

    DataFrame reset_index() const {
        DataFrame result = *this;
        result.index_.resize(result.rows_);
        for (size_t i = 0; i < result.rows_; ++i) {
            result.index_[i] = std::to_string(i);
        }
        return result;
    }

    DataFrame sample(size_t n = 1, unsigned int seed = 0) const {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<size_t> dist(0, rows_ - 1);
        std::vector<size_t> idxs(n);
        for (size_t i = 0; i < n; ++i) {
            idxs[i] = dist(gen);
        }
        return iloc(idxs);
    }

    DataFrame clip(T lower, T upper) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col.clip(lower, upper);
        }
        return result;
    }

    DataFrame ewm(T alpha) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col.ewm(alpha);
        }
        return result;
    }

    DataFrame rolling(size_t window, bool center = false) const {
        DataFrame result = *this;
        for (auto& col : result.columns_) {
            col = col.rolling(window, center);
        }
        return result;
    }

    DataFrame diff(int periods = 1) const {
        DataFrame result;
        result.engine_ = engine_;
        result.column_names_ = column_names_;
        result.rows_ = rows_;
        result.index_ = index_;
        result.rebuild_index();

        for (const auto& col : columns_) {
            result.columns_.push_back(col.diff(periods));
        }

        return result;
    }

    DataFrame round(int decimals = 0) const {
        DataFrame result = *this;
        T factor = std::pow(10.0, decimals);
        for (auto& col : result.columns_) {
            for (size_t i = 0; i < col.size(); ++i) {
                col[i] = std::round(col[i] * factor) / factor;
            }
        }
        return result;
    }

    Series<bool> isnull() const {
        Series<bool> result("isnull", engine_);
        for (const auto& col : columns_) {
            Series<bool> col_null = col.isnull();
            for (size_t i = 0; i < col_null.size(); ++i) {
                result.push_back(col_null[i]);
            }
        }
        return result;
    }

    void drop_inplace(const std::vector<size_t>& idxs) {
        std::vector<bool> keep(rows_, true);
        for (size_t idx : idxs) {
            if (idx < rows_) {
                keep[idx] = false;
            }
        }
        
        for (auto& col : columns_) {
            std::vector<T> new_data;
            for (size_t i = 0; i < col.size(); ++i) {
                if (keep[i]) {
                    new_data.push_back(col[i]);
                }
            }
            col = Series<T>(col.name(), new_data);
        }
        
        rows_ = std::count(keep.begin(), keep.end(), true);
    }

    std::shared_ptr<CpuTensorEngine<T>> engine() { return engine_; }

private:
    size_t rows_;
    std::vector<std::string> column_names_;
    std::vector<ColumnType> columns_;
    std::vector<std::string> index_;
    std::map<std::string, size_t> column_index_;
    std::shared_ptr<CpuTensorEngine<T>> engine_;

    void rebuild_index() {
        column_index_.clear();
        for (size_t i = 0; i < column_names_.size(); ++i) {
            column_index_[column_names_[i]] = i;
        }
    }
};

using DataFrameDouble = DataFrame<double>;

} // namespace ml
