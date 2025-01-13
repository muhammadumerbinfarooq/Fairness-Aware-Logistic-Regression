#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <stdexcept>

class FairLogisticRegression {
public:
    FairLogisticRegression(double learning_rate, int max_iter, double fairness_constraint)
        : learning_rate(learning_rate), max_iter(max_iter), fairness_constraint(fairness_constraint) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::vector<int>& demographic) {
        if (X.empty() || X[0].size() == 0 || y.size() != X.size() || demographic.size() != X.size()) {
            throw std::invalid_argument("Invalid input data.");
        }

        int n_samples = X.size();
        int n_features = X[0].size();
        weights.resize(n_features, 0.0);

        for (int iter = 0; iter < max_iter; ++iter) {
            // Update weights using gradient descent
            for (int i = 0; i < n_samples; ++i) {
                double prediction = predict_proba(X[i]);
                double error = y[i] - prediction;

                // Update weights
                for (int j = 0; j < n_features; ++j) {
                    weights[j] += learning_rate * error * X[i][j];
                }
            }

            // Adjust for fairness
            adjust_for_fairness(X, y, demographic);
        }
    }

    double predict_proba(const std::vector<double>& x) const {
        double linear_combination = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            linear_combination += weights[i] * x[i];
        }
        return 1.0 / (1.0 + std::exp(-linear_combination)); // Sigmoid function
    }

    int predict(const std::vector<double>& x) const {
        return predict_proba(x) >= 0.5 ? 1 : 0; // Threshold at 0.5
    }

private:
    double learning_rate;
    int max_iter;
    double fairness_constraint;
    std::vector<double> weights;

    void adjust_for_fairness(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::vector<int>& demographic) {
        // Calculate the demographic parity
        std::unordered_map<int, double> group_predictions;
        std::unordered_map<int, int> group_counts;

        for (size_t i = 0; i < X.size(); ++i) {
            int group = demographic[i];
            double prediction = predict_proba(X[i]);
            group_predictions[group] += prediction;
            group_counts[group]++;
        }

        // Calculate average predictions for each group
        for (const auto& group : group_counts) {
            group_predictions[group.first] /= group.second;
        }

        // Adjust weights based on fairness constraint
        for (const auto& group : group_counts) {
            double avg_prediction = group_predictions[group.first];
            if (avg_prediction < fairness_constraint) {
                // Adjust weights to increase predictions for this group
                for (size_t j = 0; j < weights.size(); ++j) {
                    weights[j] += learning_rate * (fairness_constraint - avg_prediction) * group_counts[group.first];
                }
            }
        }
    }
};

int main() {
    // Sample data: features, labels, and demographic groups
    std::vector<std::vector<double>> X = {
        {1.0, 2.0}, // Sample 1
        {1.5, 1.8}, // Sample 2
        {2.0, 1.0}, // Sample 3
        {1.0,  3.0}  // Sample 4
    };
    std::vector<int> y = {0, 1, 1, 0}; // Labels
    std::vector<int> demographic = {0, 1, 0, 1}; // Demographic groups

    FairLogisticRegression model(0.01, 1000, 0.5); // Learning rate, max iterations, fairness constraint
    model.fit(X, y, demographic);

    // Predictions
    for (const auto& sample : X) {
        std::cout << "Prediction: " << model.predict(sample) << std::endl;
    }

    return 0;
}
