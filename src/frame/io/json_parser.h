#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>

namespace ml {

class CSVParser2 {
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
                    row.push_back(std::nan(""));
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

class JSONParser {
public:
    using DataMap = std::map<std::string, std::vector<std::vector<double>>>;

    static DataMap parse_numeric(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        
        return parse_numeric_string(content);
    }
    
    static DataMap parse_numeric(std::istream& is) {
        std::string content((std::istreambuf_iterator<char>(is)),
                            std::istreambuf_iterator<char>());
        return parse_numeric_string(content);
    }
    
private:
    static DataMap parse_numeric_string(const std::string& json) {
        DataMap result;
        
        size_t columns_pos = json.find("\"columns\"");
        if (columns_pos != std::string::npos) {
            size_t start = json.find("[", columns_pos);
            size_t end = json.find("]", start);
            if (start != std::string::npos && end != std::string::npos) {
                std::string cols_str = json.substr(start + 1, end - start - 1);
                std::vector<std::string> columns = parse_string_array(cols_str);
                
                size_t data_pos = json.find("\"data\"");
                if (data_pos != std::string::npos) {
                    size_t data_start = json.find("[", data_pos);
                    size_t data_end = json.rfind("]");
                    
                    for (const auto& col : columns) {
                        result[col] = {};
                    }
                    
                    std::string data_str = json.substr(data_start, data_end - data_start);
                    std::vector<std::string> rows = parse_array_of_arrays(data_str);
                    
                    for (size_t r = 0; r < rows.size(); ++r) {
                        std::vector<std::string> values = parse_array(rows[r]);
                        for (size_t c = 0; c < std::min(values.size(), columns.size()); ++c) {
                            try {
                                result[columns[c]].push_back({static_cast<double>(r), std::stod(values[c])});
                            } catch (...) {
                                result[columns[c]].push_back({static_cast<double>(r), std::nan("")});
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    static std::vector<std::string> parse_string_array(const std::string& s) {
        std::vector<std::string> result;
        std::string current;
        bool in_string = false;
        
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '"' && (i == 0 || s[i-1] != '\\')) {
                in_string = !in_string;
            } else if (c == ',' && !in_string) {
                result.push_back(trim(current));
                current.clear();
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            result.push_back(trim(current));
        }
        
        return result;
    }
    
    static std::vector<std::string> parse_array(const std::string& s) {
        return parse_string_array(s);
    }
    
    static std::vector<std::string> parse_array_of_arrays(const std::string& s) {
        std::vector<std::string> result;
        std::string current;
        int depth = 0;
        
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '[') {
                if (depth == 0 && !current.empty()) {
                    current.clear();
                }
                ++depth;
            } else if (c == ']') {
                --depth;
                if (depth == 0) {
                    result.push_back(current);
                    current.clear();
                    continue;
                }
            }
            current += c;
        }
        
        return result;
    }
    
    static std::string trim(const std::string& s) {
        size_t start = 0;
        while (start < s.size() && std::isspace(s[start])) ++start;
        size_t end = s.size();
        while (end > start && std::isspace(s[end - 1])) --end;
        std::string result = s.substr(start, end - start);
        if (result.front() == '"' && result.back() == '"' && result.size() > 1) {
            result = result.substr(1, result.size() - 2);
        }
        return result;
    }
};

} // namespace ml
