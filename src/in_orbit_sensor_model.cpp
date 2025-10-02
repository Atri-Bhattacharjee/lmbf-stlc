#include "in_orbit_sensor_model.h"
#include <cmath>

namespace {
// Normalize angle to [-PI, PI]
double normalize_angle(double angle) {
    constexpr double PI = 3.14159265358979323846;
    while (angle > PI) angle -= 2.0 * PI;
    while (angle < -PI) angle += 2.0 * PI;
    return angle;
}
}

double InOrbitSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    // Check dimensions
    if (measurement.value_.size() != 6) {
        return 0.0;
    }
    if (measurement.covariance_.rows() < 6 || measurement.covariance_.cols() < 6) {
        return 0.0;
    }
    // Use full 6D state: position (x, y, z) and velocity (vx, vy, vz)
    Eigen::VectorXd particle_state = particle.state_vector.head(6);
    Eigen::VectorXd meas_value = measurement.value_.head(6);
    Eigen::MatrixXd cov = measurement.covariance_.topLeftCorner(6, 6);
    // Residual
    Eigen::VectorXd residual = meas_value - particle_state;
    // Mahalanobis distance squared
    double mahalanobis_sq = residual.transpose() * cov.inverse() * residual;
    // Normalization factor for 6D Gaussian
    constexpr double PI = 3.14159265358979323846;
    double norm_factor = std::pow(2.0 * PI, -3.0) * std::pow(cov.determinant(), -0.5);
    return norm_factor * std::exp(-0.5 * mahalanobis_sq);
}
