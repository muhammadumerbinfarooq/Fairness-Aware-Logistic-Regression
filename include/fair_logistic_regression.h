#ifndef FAIR_LOGISTIC_REGRESSION_H
#define FAIR_LOGISTIC_REGRESSION_H

#include <vector>
#include <unordered_map>
#include <stdexcept>

class FairLogisticRegression {
public:
    // Constructor to initialize the model with learning rate, max iterations, and fairness constraint
    FairLogisticRegression(double learning_rate, int max_iter, double fairness_constraint);

    // Function to fit the model to the training data
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::vector<int>& demographic);

    // Function to predict the probability of the positive class
    double predict_proba(const std::vector<double>& x) const;

    // Function to predict the class label (0 or 1)
    int predict(const std::vector<double>& x) const;

private:
    double learning_rate; // Learning rate for gradient descent
    int max_iter;         // Maximum number of iterations for training
    double fairness_constraint; // Fairness constraint for demographic parity
    std::vector<double> weights; // Weights for the logistic regression model

    // Function to adjust weights for fairness after each iteration
    void adjust_for_fairness(const std::vector<std::vector<double>>& X, const std::vector<int>& y, const std::vector<int>& demographic);
};

#endif // FAIR_LOGISTIC_REGRESSION_H
