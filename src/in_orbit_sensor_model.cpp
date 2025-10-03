#include "in_orbit_sensor_model.h"
#include <cmath>

double InOrbitSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    // Check dimensions
    if (measurement.value_.size() != 6) {
        return 0.0;
    }
    if (measurement.covariance_.rows() != 6 || measurement.covariance_.cols() != 6) {
        return 0.0;
    }
    // Use full 6D state: position (x, y, z), velocity (vx, vy, vz)
    Eigen::VectorXd particle_state = particle.state_vector;
    Eigen::VectorXd meas_value = measurement.value_;
    Eigen::MatrixXd cov = measurement.covariance_;
    // Residual
    Eigen::VectorXd residual = meas_value - particle_state;
    // Mahalanobis distance squared
    double mahalanobis_sq = residual.transpose() * cov.inverse() * residual;
    // Normalization factor for 6D Gaussian
    constexpr double PI = 3.14159265358979323846;
    double norm_factor = std::pow(2.0 * PI, -3.0) * std::pow(cov.determinant(), -0.5);
    return norm_factor * std::exp(-0.5 * mahalanobis_sq);
}
