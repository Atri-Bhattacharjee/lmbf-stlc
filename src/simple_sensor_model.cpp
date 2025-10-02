#include "simple_sensor_model.h"
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- BEGIN DEFINITIVE REPLACEMENT ---
double SimpleSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    const Eigen::VectorXd& state = particle.state_vector;
    const Eigen::VectorXd& meas = measurement.value_;
    const Eigen::MatrixXd& cov = measurement.covariance_;

    // 1. Verify Input Sizes
    if (state.size() < 6 || meas.size() < 6 || cov.rows() < 6 || cov.cols() < 6) {
        return 0.0;
    }

    // 2. Extract 6D Subvectors
    Eigen::VectorXd state_6d = state.head(6);
    Eigen::VectorXd meas_6d = meas.head(6);
    Eigen::MatrixXd cov_6d = cov.topLeftCorner(6, 6);

    // 3. Calculate Residual
    Eigen::VectorXd residual = meas_6d - state_6d;

    // 4. Calculate Inverse Covariance
    Eigen::MatrixXd cov_inv = cov_6d.inverse();
    
    // 5. Calculate Mahalanobis Distance Squared
    double maha_sq = residual.transpose() * cov_inv * residual;

    // 6. Calculate Exponential Term
    double exp_val = std::exp(-0.5 * maha_sq);

    // 7. Calculate Normalization Factor (for completeness, though not strictly needed)
    const double PI = 3.14159265358979323846;
    double norm_factor = std::pow(2.0 * PI, -3.0) * std::pow(cov_6d.determinant(), -0.5);
    
    double final_likelihood = norm_factor * exp_val;

    // Add the floor just in case, though the issue is upstream
    const double likelihood_floor = 1e-12;
    return std::max(final_likelihood, likelihood_floor);
}
// --- END DEFINITIVE REPLACEMENT ---