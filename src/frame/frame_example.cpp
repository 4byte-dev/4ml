#include <iostream>
#include <fstream>
#include <frame/frame.h>
#include <frame/series.h>
#include <frame/stats.h>
#include <frame/ml.h>

int main() {

    ml::DataFrame<double> df;
    df.add_column("A", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    df.add_column("B", std::vector<double>{10.0, 20.0, 30.0, 40.0, 50.0});
    df.add_column("C", std::vector<double>{100.0, 200.0, 300.0, 400.0, 500.0});
    
    std::cout << "=== Original DataFrame ===\n" << df.to_string() << "\n";
    
    std::cout << "=== Statistics ===\n";
    std::cout << "Mean of A: " << df.col("A").mean() << "\n";
    std::cout << "Sum of B: " << df.col("B").sum() << "\n";
    std::cout << "Std of C: " << df.col("C").std() << "\n";
    std::cout << "Min of A: " << df.col("A").min() << "\n";
    std::cout << "Max of C: " << df.col("C").max() << "\n";
    std::cout << "Median of B: " << df.col("B").median() << "\n";
    std::cout << "Quantile 25%: " << df.col("A").quantile(0.25) << "\n";
    std::cout << "Quantile 75%: " << df.col("A").quantile(0.75) << "\n";
    
    std::cout << "\n=== Operations ===\n";
    auto df2 = df * 2.0;
    std::cout << "DataFrame * 2:\n" << df2.to_string() << "\n";
    
    auto df3 = df + df2;
    std::cout << "DataFrame + DataFrame*2:\n" << df3.to_string() << "\n";
    
    std::cout << "\n=== Indexing ===\n";
    std::cout << "Head(3):\n" << df.head(3).to_string() << "\n";
    std::cout << "Tail(2):\n" << df.tail(2).to_string() << "\n";
    
    std::cout << "\n=== Sorting ===\n";
    auto sorted_desc = df.sort_values("A", false);
    std::cout << "Sorted by A (descending):\n" << sorted_desc.to_string() << "\n";
    
    std::cout << "\n=== Stats Module ===\n";
    auto data = df.col("A").to_vector();
    std::cout << "Variance: " << ml::stats::variance(data) << "\n";
    std::cout << "Skewness: " << ml::stats::skewness(data) << "\n";
    std::cout << "Kurtosis: " << ml::stats::kurtosis(data) << "\n";
    std::cout << "Median: " << ml::stats::median(data) << "\n";
    
    std::cout << "\n=== Normalization ===\n";
    auto normalized = ml::stats::normalize(data);
    std::cout << "Normalized A: [";
    for (size_t i = 0; i < normalized.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << normalized[i];
    }
    std::cout << "]\n";
    
    std::cout << "\n=== Standardization ===\n";
    auto standardized = ml::stats::standardize(data);
    std::cout << "Standardized A: [";
    for (size_t i = 0; i < standardized.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << standardized[i];
    }
    std::cout << "]\n";
    
    std::cout << "\n=== Rolling Window ===\n";
    auto rolled = df.col("A").rolling(2);
    std::cout << "Rolling mean (window=2):\n";
    rolled.print();
    
    std::cout << "\n=== Cumulative Operations ===\n";
    auto cumsum = df.col("A").cumsum();
    std::cout << "Cumulative sum of A:\n";
    cumsum.print();
    
    auto cummax = df.col("A").cummax();
    std::cout << "Cumulative max of A:\n";
    cummax.print();
    
    std::cout << "\n=== Exponentially Weighted ===\n";
    auto ewm = df.col("A").ewm(0.5);
    std::cout << "EWM (alpha=0.5):\n";
    ewm.print();
    
    std::cout << "\n=== Diff ===\n";
    auto diff = df.col("A").diff();
    std::cout << "Diff of A:\n";
    diff.print();
    
    std::cout << "\n=== DataFrame Info ===\n";
    df.info();
    
    std::cout << "\n=== ML Module Examples ===\n";
    
    ml::DataFrame<double> X_train;
    X_train.add_column("Feature1", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    X_train.add_column("Feature2", std::vector<double>{2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0});
    
    ml::DataFrame<double> y_train;
    y_train.add_column("Target", std::vector<double>{3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0});
    
    std::cout << "\n--- Training Data ---\n";
    std::cout << "X_train:\n" << X_train.to_string() << "\n";
    std::cout << "y_train:\n" << y_train.to_string() << "\n";
    
    std::cout << "\n--- Train-Test Split ---\n";
    auto split = ml::ml_ops::train_test_split(X_train, y_train, 0.2, 42);
    std::cout << "X_train split: " << split.X_train.rows() << " rows\n";
    std::cout << "X_test split: " << split.X_test.rows() << " rows\n";
    
    std::cout << "\n--- Linear Regression ---\n";
    auto [weights, bias] = ml::ml_ops::linear_regression_fit(X_train, y_train.col("Target"));
    std::cout << "Weights: [";
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << weights[i];
    }
    std::cout << "]\n";
    std::cout << "Bias: " << bias << "\n";
    
    auto predictions = ml::ml_ops::linear_regression_predict(X_train, weights, bias);
    std::cout << "Predictions:\n" << predictions.to_string() << "\n";
    
    std::cout << "\n--- Evaluation Metrics ---\n";
    ml::Series<double> y_true("Y", std::vector<double>{3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0});
    ml::Series<double> y_pred("Y", std::vector<double>{3.1, 6.2, 8.9, 12.3, 14.8, 17.9, 21.2, 23.8, 27.1, 30.2});
    
    double mse = ml::ml_ops::mean_squared_error(y_true, y_pred);
    double rmse = ml::ml_ops::root_mean_squared_error(y_true, y_pred);
    double mae = ml::ml_ops::mean_absolute_error(y_true, y_pred);
    double r2 = ml::ml_ops::r2_score(y_true, y_pred);
    
    std::cout << "MSE: " << mse << "\n";
    std::cout << "RMSE: " << rmse << "\n";
    std::cout << "MAE: " << mae << "\n";
    std::cout << "R2 Score: " << r2 << "\n";
    
    std::cout << "\n--- Min-Max Scaling ---\n";
    auto X_scaled = ml::ml_ops::min_max_scaler(X_train);
    std::cout << "Min-Max Scaled:\n" << X_scaled.to_string() << "\n";
    
    std::cout << "\n--- Standard Scaling ---\n";
    auto X_standardized = ml::ml_ops::standard_scaler(X_train);
    std::cout << "Standardized:\n" << X_standardized.to_string() << "\n";
    
    std::cout << "\n--- Polynomial Features (degree=2) ---\n";
    ml::DataFrame<double> X_poly;
    X_poly.add_column("F1", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    auto poly_features = ml::ml_ops::polynomial_features(X_poly, 2);
    std::cout << "Polynomial Features:\n" << poly_features.to_string() << "\n";
    
    std::cout << "\n--- Cross Terms ---\n";
    ml::DataFrame<double> X_cross;
    X_cross.add_column("A", std::vector<double>{1.0, 2.0, 3.0});
    X_cross.add_column("B", std::vector<double>{4.0, 5.0, 6.0});
    auto cross_terms = ml::ml_ops::cross_terms(X_cross);
    std::cout << "Cross Terms:\n" << cross_terms.to_string() << "\n";
    
    std::cout << "\n--- K-Means Clustering ---\n";
    std::vector<double> cluster_data = {1.0, 1.5, 2.0, 10.0, 10.5, 11.0, 20.0, 20.5, 21.0};
    auto centroids = ml::ml_ops::k_means_clustering(cluster_data, 3, 100, 42);
    std::cout << "K-Means Centroids (k=3): [";
    for (size_t i = 0; i < centroids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << centroids[i];
    }
    std::cout << "]\n";
    
    std::cout << "\n--- Label Encoding ---\n";
    std::vector<double> labels = {1.0, 0.0, 1.0, 2.0, 0.0, 1.0};
    auto encoded = ml::ml_ops::label_encode(labels);
    std::cout << "Original: [1, 0, 1, 2, 0, 1]\n";
    std::cout << "Encoded: [";
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << encoded[i];
    }
    std::cout << "]\n";
    
    std::cout << "\n--- Add Bias Term ---\n";
    auto X_with_bias = ml::ml_ops::add_bias_term(X_train);
    std::cout << "With Bias Column:\n" << X_with_bias.to_string() << "\n";
    
    std::cout << "\n--- Logistic Regression ---\n";
    std::vector<double> features = {1.0, 2.0, 3.0};
    std::vector<double> lr_weights = {0.5, -0.3, 0.8};
    double lr_bias = -0.5;
    double prob = ml::ml_ops::logistic_regression_predict(features, lr_weights, lr_bias);
    std::cout << "Probability: " << prob << "\n";
    
    std::cout << "\n--- Classification Metrics ---\n";
    ml::Series<double> y_true_cls("Y", std::vector<double>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0});
    ml::Series<double> y_pred_cls("Y", std::vector<double>{0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    double acc = ml::ml_ops::accuracy_score(y_true_cls, y_pred_cls);
    std::cout << "Accuracy: " << acc << "\n";
    
    std::cout << "\n=== Additional Stats Functions ===\n";
    std::vector<double> sample_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    auto desc = ml::stats::describe(sample_data);
    std::cout << "Describe:\n";
    for (const auto& [key, val] : desc) {
        std::cout << "  " << key << ": " << val << "\n";
    }
    
    std::cout << "\n=== Covariance & Correlation ===\n";
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.0, 4.0, 6.0, 8.0, 10.0};
    std::cout << "Covariance: " << ml::stats::covariance(x, y) << "\n";
    std::cout << "Correlation: " << ml::stats::correlation(x, y) << "\n";
    
    std::cout << "\n=== DataFrame Concatenation ===\n";
    ml::DataFrame<double> df_a;
    df_a.add_column("X", std::vector<double>{1.0, 2.0, 3.0});
    
    ml::DataFrame<double> df_b;
    df_b.add_column("X", std::vector<double>{4.0, 5.0, 6.0});
    
    auto concatenated = df_a.concat(df_b);
    std::cout << "Concatenated:\n" << concatenated.to_string() << "\n";
    
    std::cout << "\n=== CSV Output ===\n";
    std::cout << "Writing to /tmp/output.csv...\n";
    df.to_csv("/tmp/output.csv");
    std::cout << "Done! Contents:\n";
    std::ifstream file("/tmp/output.csv");
    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }
    
    return 0;
}
