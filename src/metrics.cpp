#include "metrics.h"
#include "assignment.h"
#include <cmath>
#include <algorithm>

// Helper: Compute mean ECI state (position+velocity) of a track
static Eigen::VectorXd mean_state(const Track& track) {
    const auto& particles = track.particles();
    if (particles.empty()) return Eigen::VectorXd::Zero(6);
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(6);
    double total_weight = 0.0;
    for (const auto& p : particles) {
        sum += p.state_vector.head(6) * p.weight;
        total_weight += p.weight;
    }
    if (total_weight == 0.0) return Eigen::VectorXd::Zero(6);
    return sum / total_weight;
}

// Helper: Compute mean state of a track's particles (returns 7D, but for metrics use only first 6)
Eigen::VectorXd mean_state6d(const std::vector<Particle>& particles) {
    if (particles.empty()) return Eigen::VectorXd();
    int dim = particles[0].state_vector.size();
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
    double total_weight = 0.0;
    for (const auto& p : particles) {
        if (p.state_vector.size() != dim) {
            return Eigen::VectorXd();
        }
        mean += p.state_vector * p.weight;
        total_weight += p.weight;
    }
    if (total_weight > 0.0) mean /= total_weight;
    // Truncate to first 6 elements for metric comparison
    return mean.head(6);
}

double calculate_ospa_distance(const std::vector<Track>& tracks,
                               const std::vector<Eigen::VectorXd>& ground_truths,
                               double cutoff) {
    // Check shapes
    for (const auto& t : tracks) {
        Eigen::VectorXd mean6d = mean_state6d(t.particles());
        if (mean6d.size() == 0) {
            return cutoff;
        }
        if (!ground_truths.empty() && mean6d.size() != ground_truths[0].size()) {
            return cutoff;
        }
    }
    size_t m = tracks.size();
    size_t n = ground_truths.size();
    if (n == 0) return 0.0;
    if (m == 0) return cutoff;
    bool tracks_are_smaller = m <= n;
    size_t rows = tracks_are_smaller ? m : n;
    size_t cols = tracks_are_smaller ? n : m;
    Eigen::MatrixXd dist_matrix(rows, cols);
    if (tracks_are_smaller) {
        for (size_t i = 0; i < m; ++i) {
            Eigen::VectorXd track_state = mean_state(tracks[i]);
            for (size_t j = 0; j < n; ++j) {
                dist_matrix(i, j) = std::min((track_state - ground_truths[j]).norm(), cutoff);
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            Eigen::VectorXd truth_state = ground_truths[i];
            for (size_t j = 0; j < m; ++j) {
                dist_matrix(i, j) = std::min((mean_state(tracks[j]) - truth_state).norm(), cutoff);
            }
        }
    }
    auto hyps = solve_assignment(dist_matrix, 1);
    double assignment_sum = 0.0;
    if (!hyps.empty()) {
        const auto& assoc_vec = hyps[0].associations;
        for (size_t i = 0; i < rows; ++i) {
            int j = (i < assoc_vec.size()) ? assoc_vec[i] : -1;
            if (j != -1 && j >= 0 && j < (int)cols)
                assignment_sum += dist_matrix(i, j);
            else
                assignment_sum += cutoff;
        }
    } else {
        assignment_sum = rows * cutoff;
    }
    double cardinality_error = cutoff * std::abs((int)m - (int)n);
    double ospa = (assignment_sum + cardinality_error) / std::max(m, n);
    return ospa;
}
