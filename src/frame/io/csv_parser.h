#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace ml {

class CSVParser {
public:
    static std::vector<std::string> parse_line(const std::string& line, char delimiter = ',') {
        std::vector<std::string> result;
        std::string current;
        bool in_quotes = false;
        
        for (size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            
            if (c == '"') {
                if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                    current += '"';
                    ++i;
                } else {
                    in_quotes = !in_quotes;
                }
            } else if (c == delimiter && !in_quotes) {
                result.push_back(trim(current));
                current.clear();
            } else {
                current += c;
            }
        }
        
        result.push_back(trim(current));
        return result;
    }

    static std::vector<std::vector<std::string>> parse(const std::string& filename, char delimiter = ',') {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        return parse(file, delimiter);
    }

    static std::vector<std::vector<std::string>> parse(std::istream& is, char delimiter = ',') {
        std::vector<std::vector<std::string>> result;
        std::string line;
        
        while (std::getline(is, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty()) {
                result.push_back(parse_line(line, delimiter));
            }
        }
        
        return result;
    }

    static std::vector<std::vector<double>> parse_numeric(const std::string& filename, 
                                                           char delimiter = ',',
                                                           size_t skip_rows = 0) {
        auto data = parse(filename, delimiter);
        std::vector<std::vector<double>> result;
        
        size_t start = std::min(skip_rows, data.size());
        for (size_t i = start; i < data.size(); ++i) {
            std::vector<double> row;
            for (const auto& cell : data[i]) {
                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    row.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            }
            result.push_back(row);
        }
        
        return result;
    }

    static void write(const std::string& filename, 
                     const std::vector<std::string>& headers,
                     const std::vector<std::vector<double>>& data,
                     char delimiter = ',') {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        for (size_t i = 0; i < headers.size(); ++i) {
            if (i > 0) file << delimiter;
            file << headers[i];
        }
        file << "\n";
        
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) file << delimiter;
                file << row[i];
            }
            file << "\n";
        }
    }

    static void write_line(std::ostream& os, 
                          const std::vector<std::string>& values,
                          char delimiter = ',') {
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) os << delimiter;
            if (values[i].find(delimiter) != std::string::npos || 
                values[i].find('"') != std::string::npos) {
                os << "\"" << values[i] << "\"";
            } else {
                os << values[i];
            }
        }
    }

private:
    static std::string trim(const std::string& s) {
        size_t start = 0;
        while (start < s.size() && std::isspace(s[start])) ++start;
        size_t end = s.size();
        while (end > start && std::isspace(s[end - 1])) --end;
        return s.substr(start, end - start);
    }
};

} // namespace ml
