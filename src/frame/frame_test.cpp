#include <gtest/gtest.h>
#include <frame/frame.h>
#include <frame/series.h>
#include <frame/stats.h>
#include <frame/ml.h>
#include <vector>
#include <sstream>
#include <cmath>

using namespace ml;

class DataFrameTest : public ::testing::Test {
protected:
    void SetUp() override {
        df1.add_column("A", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
        df1.add_column("B", std::vector<double>{10.0, 20.0, 30.0, 40.0, 50.0});
    }
    
    DataFrame<double> df1;
};

TEST_F(DataFrameTest, ConstructorDefault) {
    DataFrame<double> df;
    EXPECT_EQ(df.rows(), 0);
    EXPECT_EQ(df.cols(), 0);
    EXPECT_TRUE(df.empty());
}

TEST_F(DataFrameTest, ConstructorWithSize) {
    DataFrame<double> df(3, 2);
    EXPECT_EQ(df.rows(), 3);
    EXPECT_EQ(df.cols(), 2);
}

TEST_F(DataFrameTest, AddColumn) {
    DataFrame<double> df;
    df.add_column("X", std::vector<double>{1.0, 2.0, 3.0});
    EXPECT_EQ(df.cols(), 1);
    EXPECT_EQ(df.rows(), 3);
    EXPECT_TRUE(df.contains("X"));
}

TEST_F(DataFrameTest, RemoveColumn) {
    df1.remove_column("A");
    EXPECT_EQ(df1.cols(), 1);
    EXPECT_FALSE(df1.contains("A"));
}

TEST_F(DataFrameTest, ColAccessor) {
    auto& col = df1.col("A");
    EXPECT_EQ(col.size(), 5);
    EXPECT_EQ(col[0], 1.0);
}

TEST_F(DataFrameTest, Head) {
    auto head = df1.head(3);
    EXPECT_EQ(head.rows(), 3);
}

TEST_F(DataFrameTest, Tail) {
    auto tail = df1.tail(2);
    EXPECT_EQ(tail.rows(), 2);
}

TEST_F(DataFrameTest, IlocSingle) {
    auto row = df1.iloc(0);
    EXPECT_EQ(row.rows(), 1);
}

TEST_F(DataFrameTest, IlocRange) {
    auto rows = df1.iloc(1, 3);
    EXPECT_EQ(rows.rows(), 2);
}

TEST_F(DataFrameTest, SortValues) {
    auto sorted = df1.sort_values("A", false);
    EXPECT_EQ(sorted.col("A")[0], 5.0);
}

TEST_F(DataFrameTest, DropNa) {
    DataFrame<double> df;
    df.add_column("X", std::vector<double>{1.0, 2.0, 3.0});
    auto dropped = df.dropna();
    EXPECT_EQ(dropped.rows(), 3);
}

TEST_F(DataFrameTest, FillNa) {
    DataFrame<double> df;
    df.add_column("X", std::vector<double>{1.0, 2.0, 3.0});
    auto filled = df.fillna(5.0);
    EXPECT_EQ(filled.col("X")[0], 1.0);
}

TEST_F(DataFrameTest, IndexOf) {
    auto idx = df1.index_of("A");
    EXPECT_TRUE(idx.has_value());
    EXPECT_EQ(idx.value(), 0);
}

TEST_F(DataFrameTest, Contains) {
    EXPECT_TRUE(df1.contains("A"));
    EXPECT_FALSE(df1.contains("Z"));
}

TEST_F(DataFrameTest, OperatorPlusScalar) {
    auto result = df1 + 1.0;
    EXPECT_EQ(result.col("A")[0], 2.0);
}

TEST_F(DataFrameTest, OperatorMinusScalar) {
    auto result = df1 - 1.0;
    EXPECT_EQ(result.col("A")[0], 0.0);
}

TEST_F(DataFrameTest, OperatorMultiplyScalar) {
    auto result = df1 * 2.0;
    EXPECT_EQ(result.col("A")[0], 2.0);
}

TEST_F(DataFrameTest, OperatorDivideScalar) {
    auto result = df1 / 2.0;
    EXPECT_EQ(result.col("A")[0], 0.5);
}

TEST_F(DataFrameTest, OperatorPlusDataFrame) {
    auto result = df1 + df1;
    EXPECT_EQ(result.col("A")[0], 2.0);
}

TEST_F(DataFrameTest, Cumsum) {
    auto result = df1.cumsum();
    EXPECT_EQ(result.col("A")[0], 1.0);
    EXPECT_EQ(result.col("A")[1], 3.0);
}

TEST_F(DataFrameTest, Diff) {
    auto result = df1.diff();
    EXPECT_EQ(result.col("A")[1], 1.0);
}

TEST_F(DataFrameTest, Describe) {
    auto result = df1.describe();
    EXPECT_GT(result.rows(), 0);
}

TEST_F(DataFrameTest, ToCsv) {
    std::ostringstream oss;
    df1.to_csv(oss);
    std::string output = oss.str();
    EXPECT_NE(output.find("A"), std::string::npos);
}

TEST_F(DataFrameTest, ToString) {
    std::string str = df1.to_string();
    EXPECT_FALSE(str.empty());
}

TEST_F(DataFrameTest, Info) {
    testing::internal::CaptureStdout();
    df1.info();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_FALSE(output.empty());
}

TEST_F(DataFrameTest, MemoryUsage) {
    size_t mem = df1.memory_usage();
    EXPECT_GT(mem, 0);
}

class SeriesTest : public ::testing::Test {
protected:
    void SetUp() override {
        series1 = Series<double>("Test", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    }
    
    Series<double> series1;
};

TEST_F(SeriesTest, Constructor) {
    EXPECT_EQ(series1.size(), 5);
    EXPECT_EQ(series1.name(), "Test");
}

TEST_F(SeriesTest, OperatorIndex) {
    EXPECT_EQ(series1[0], 1.0);
    EXPECT_EQ(series1[4], 5.0);
}

TEST_F(SeriesTest, Sum) {
    EXPECT_EQ(series1.sum(), 15.0);
}

TEST_F(SeriesTest, Mean) {
    EXPECT_EQ(series1.mean(), 3.0);
}

TEST_F(SeriesTest, Min) {
    EXPECT_EQ(series1.min(), 1.0);
}

TEST_F(SeriesTest, Max) {
    EXPECT_EQ(series1.max(), 5.0);
}

TEST_F(SeriesTest, Std) {
    double std_val = series1.std();
    EXPECT_GT(std_val, 0);
}

TEST_F(SeriesTest, Median) {
    EXPECT_EQ(series1.median(), 3.0);
}

TEST_F(SeriesTest, Quantile) {
    EXPECT_EQ(series1.quantile(0.5), 3.0);
}

TEST_F(SeriesTest, Idxmin) {
    EXPECT_EQ(series1.idxmin(), 0);
}

TEST_F(SeriesTest, Idxmax) {
    EXPECT_EQ(series1.idxmax(), 4);
}

TEST_F(SeriesTest, Head) {
    auto result = series1.head(3);
    EXPECT_EQ(result.size(), 3);
}

TEST_F(SeriesTest, Tail) {
    auto result = series1.tail(2);
    EXPECT_EQ(result.size(), 2);
}

TEST_F(SeriesTest, Diff) {
    auto result = series1.diff();
    EXPECT_EQ(result[1], 1.0);
}

TEST_F(SeriesTest, Cumsum) {
    auto result = series1.cumsum();
    EXPECT_EQ(result[0], 1.0);
    EXPECT_EQ(result[1], 3.0);
}

TEST_F(SeriesTest, OperatorPlusSeries) {
    auto result = series1 + series1;
    EXPECT_EQ(result[0], 2.0);
}

TEST_F(SeriesTest, OperatorPlusScalar) {
    auto result = series1 + 1.0;
    EXPECT_EQ(result[0], 2.0);
}

TEST_F(SeriesTest, OperatorMinusScalar) {
    auto result = series1 - 1.0;
    EXPECT_EQ(result[0], 0.0);
}

TEST_F(SeriesTest, OperatorMultiplyScalar) {
    auto result = series1 * 2.0;
    EXPECT_EQ(result[0], 2.0);
}

TEST_F(SeriesTest, OperatorDivideScalar) {
    auto result = series1 / 2.0;
    EXPECT_EQ(result[0], 0.5);
}

TEST_F(SeriesTest, SortAscending) {
    auto result = series1.sort(true);
    EXPECT_EQ(result[0], 1.0);
}

TEST_F(SeriesTest, SortDescending) {
    auto result = series1.sort(false);
    EXPECT_EQ(result[0], 5.0);
}

TEST_F(SeriesTest, ToVector) {
    auto vec = series1.to_vector();
    EXPECT_EQ(vec.size(), 5);
    EXPECT_EQ(vec[0], 1.0);
}

TEST_F(SeriesTest, Dot) {
    Series<double> other("Other", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    EXPECT_EQ(series1.dot(other), 55.0);
}

class StatsTest : public ::testing::Test {
protected:
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
};

TEST_F(StatsTest, Mean) {
    EXPECT_DOUBLE_EQ(stats::mean(data), 3.0);
}

TEST_F(StatsTest, Variance) {
    EXPECT_GT(stats::variance(data), 0);
}

TEST_F(StatsTest, StdDev) {
    EXPECT_GT(stats::std_dev(data), 0);
}

TEST_F(StatsTest, Median) {
    EXPECT_DOUBLE_EQ(stats::median(data), 3.0);
}

TEST_F(StatsTest, Percentile) {
    EXPECT_DOUBLE_EQ(stats::percentile(data, 0.5), 3.0);
}

TEST_F(StatsTest, Normalize) {
    auto result = stats::normalize(data);
    EXPECT_EQ(result[0], 0.0);
    EXPECT_EQ(result[4], 1.0);
}

TEST_F(StatsTest, Standardize) {
    auto result = stats::standardize(data);
    EXPECT_LT(result[0], 0);
}

TEST_F(StatsTest, Describe) {
    auto result = stats::describe(data);
    EXPECT_EQ(result["count"], 5.0);
    EXPECT_EQ(result["mean"], 3.0);
}

class MLTest : public ::testing::Test {
protected:
    void SetUp() override {
        X.add_column("F1", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
        X.add_column("F2", std::vector<double>{10.0, 20.0, 30.0, 40.0, 50.0});
        y.add_column("Y", std::vector<double>{2.0, 4.0, 6.0, 8.0, 10.0});
    }
    
    DataFrame<double> X;
    DataFrame<double> y;
};

TEST_F(MLTest, TrainTestSplit) {
    auto split = ml_ops::train_test_split(X, y, 0.2, 42);
    EXPECT_GT(split.X_train.rows(), 0);
    EXPECT_GT(split.X_test.rows(), 0);
}

TEST_F(MLTest, MinMaxScaler) {
    auto result = ml_ops::min_max_scaler(X);
    EXPECT_EQ(result.cols(), X.cols());
}

TEST_F(MLTest, StandardScaler) {
    auto result = ml_ops::standard_scaler(X);
    EXPECT_EQ(result.cols(), X.cols());
}

TEST_F(MLTest, LinearRegressionFit) {
    auto [weights, bias] = ml_ops::linear_regression_fit(X, y.col("Y"));
    EXPECT_EQ(weights.size(), X.cols());
}

TEST_F(MLTest, MeanSquaredError) {
    Series<double> y_true("Y", std::vector<double>{1.0, 2.0, 3.0});
    Series<double> y_pred("Y", std::vector<double>{1.1, 2.1, 2.9});
    double mse = ml_ops::mean_squared_error(y_true, y_pred);
    EXPECT_GT(mse, 0);
}

TEST_F(MLTest, R2Score) {
    Series<double> y_true("Y", std::vector<double>{1.0, 2.0, 3.0});
    Series<double> y_pred("Y", std::vector<double>{1.1, 2.1, 2.9});
    double r2 = ml_ops::r2_score(y_true, y_pred);
    EXPECT_LT(r2, 1.0);
}

TEST_F(MLTest, AccuracyScore) {
    Series<double> y_true("Y", std::vector<double>{0.0, 1.0, 0.0, 1.0});
    Series<double> y_pred("Y", std::vector<double>{0.0, 1.0, 1.0, 1.0});
    double acc = ml_ops::accuracy_score(y_true, y_pred);
    EXPECT_EQ(acc, 0.75);
}

class CSVParserTest : public ::testing::Test {};

TEST_F(CSVParserTest, ParseLine) {
    auto result = CSVParser::parse_line("a,b,c", ',');
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
}

TEST_F(CSVParserTest, ParseLineWithQuotes) {
    auto result = CSVParser::parse_line("\"a,b\",c", ',');
    EXPECT_EQ(result.size(), 2);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
