#pragma once

#include <vector>
#include <Eigen/Dense>

struct Hypothesis {
    std::vector<int> associations; // track index -> measurement index (-1 for miss)
    double weight; // log-likelihood or cost
    Hypothesis() : associations(), weight(0.0) {}
    Hypothesis(const std::vector<int>& assoc, double w) : associations(assoc), weight(w) {}
};

std::vector<Hypothesis> solve_assignment(const Eigen::MatrixXd& cost_matrix, int k_best);
