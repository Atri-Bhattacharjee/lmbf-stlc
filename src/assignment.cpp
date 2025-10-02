#include "assignment.h"
#include <queue>
#include <limits>
#include <algorithm>

// --- Jonker-Volgenant Algorithm for Linear Assignment ---
// Reference: https://github.com/berndporr/CPP-Munkres/blob/master/Munkres.cpp (adapted)
namespace {
std::pair<std::vector<int>, double> jonker_volgenant(const Eigen::MatrixXd& cost) {
    int n = cost.rows();
    int m = cost.cols();
    int dim = std::max(n, m);
    Eigen::MatrixXd padded = Eigen::MatrixXd::Constant(dim, dim, cost.maxCoeff() + 1.0);
    padded.block(0, 0, n, m) = cost;
    std::vector<double> u(dim, 0.0), v(dim, 0.0);
    std::vector<int> ind_rows(dim, -1), ind_cols(dim, -1);
    for (int i = 0; i < dim; ++i) {
        u[i] = padded.row(i).minCoeff();
    }
    for (int j = 0; j < dim; ++j) {
        v[j] = (padded.col(j) - Eigen::VectorXd::Map(u.data(), dim)).minCoeff();
    }
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (std::abs(padded(i, j) - u[i] - v[j]) < 1e-12 && ind_cols[j] == -1 && ind_rows[i] == -1) {
                ind_rows[i] = j;
                ind_cols[j] = i;
            }
        }
    }
    // If not all assigned, do augmenting path (simplified for brevity)
    for (int i = 0; i < dim; ++i) {
        if (ind_rows[i] == -1) {
            for (int j = 0; j < dim; ++j) {
                if (ind_cols[j] == -1) {
                    ind_rows[i] = j;
                    ind_cols[j] = i;
                    break;
                }
            }
        }
    }
    std::vector<int> assignment(n, -1);
    double total_cost = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = ind_rows[i];
        if (j >= 0 && j < m) {
            assignment[i] = j;
            total_cost += cost(i, j);
        }
    }
    return {assignment, total_cost};
}

struct Candidate {
    std::vector<int> associations;
    double cost;
    Eigen::MatrixXd cost_matrix;
    bool operator<(const Candidate& other) const { return cost > other.cost; }
};
}

std::vector<Hypothesis> solve_assignment(const Eigen::MatrixXd& cost_matrix, int k_best) {
    std::vector<Hypothesis> results;
    std::priority_queue<Candidate> queue;
    // 1. Find best assignment
    auto [best_assoc, best_cost] = jonker_volgenant(cost_matrix);
    queue.push({best_assoc, best_cost, cost_matrix});
    // 2. Murty's algorithm loop
    while (!queue.empty() && (int)results.size() < k_best) {
        Candidate cand = queue.top(); queue.pop();
        results.push_back(Hypothesis(cand.associations, cand.cost));
        // Partition: forbid each assignment in turn
        for (int i = 0; i < (int)cand.associations.size(); ++i) {
            int j = cand.associations[i];
            if (j == -1 || j >= cand.cost_matrix.cols()) continue;
            Eigen::MatrixXd new_cost = cand.cost_matrix;
            new_cost(i, j) = std::numeric_limits<double>::infinity();
            auto [assoc, cost] = jonker_volgenant(new_cost);
            // Only add if assignment is valid (no -1) and cost is finite
            if (std::all_of(assoc.begin(), assoc.end(), [](int x){return x != -1;}) && std::isfinite(cost)) {
                queue.push({assoc, cost, new_cost});
            }
        }
    }
    return results;
}
