#include "in_orbit_sensor_model.h"
#include <cmath>
#include <tuple>

double InOrbitSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    // Check dimensions
    if (measurement.value_.size() != 6) {
        return 0.0;
    }
    if (measurement.covariance_.rows() != 6 || measurement.covariance_.cols() != 6) {
        return 0.0;
    }
    
    // For the refactored EKF version, we calculate the likelihood using
    // the innovation (y - h(x)) and the innovation covariance (S = HPH' + R)
    Eigen::VectorXd predicted_measurement = predict_measurement(particle.mean);
    Eigen::VectorXd meas_value = measurement.value_;
    Eigen::MatrixXd meas_cov = measurement.covariance_;
    
    // Compute innovation
    Eigen::VectorXd innovation = meas_value - predicted_measurement;
    
    // Compute measurement Jacobian
    Eigen::MatrixXd H = get_measurement_jacobian(particle.mean);
    
    // Compute innovation covariance
    Eigen::MatrixXd S = H * particle.covariance * H.transpose() + meas_cov;
    
    // Compute Mahalanobis distance squared
    double mahalanobis_sq = innovation.transpose() * S.inverse() * innovation;
    
    // Normalization factor for 6D Gaussian
    constexpr double PI = 3.14159265358979323846;
    double norm_factor = std::pow(2.0 * PI, -3.0) * std::pow(S.determinant(), -0.5);
    
    // Compute likelihood
    return norm_factor * std::exp(-0.5 * mahalanobis_sq);
}

Eigen::VectorXd InOrbitSensorModel::predict_measurement(const Eigen::VectorXd& state) const {
    // For this sensor model, the measurement function is simple:
    // we directly observe the state (position and velocity)
    return state;
}

Eigen::MatrixXd InOrbitSensorModel::get_measurement_jacobian(const Eigen::VectorXd& state) const {
    // For this sensor model, the Jacobian is just the identity matrix
    // since we directly observe the state
    int state_dim = state.size();
    return Eigen::MatrixXd::Identity(state_dim, state_dim);
}

std::tuple<Particle, double> InOrbitSensorModel::ekf_update(const Particle& particle, 
                                                         const Measurement& measurement) const {
    // Create the updated particle
    Particle updated_particle;
    
    // Get measurement prediction
    Eigen::VectorXd predicted_measurement = predict_measurement(particle.mean);
    
    // Get measurement Jacobian
    Eigen::MatrixXd H = get_measurement_jacobian(particle.mean);
    
    // Compute innovation
    Eigen::VectorXd innovation = measurement.value_ - predicted_measurement;
    
    // Compute innovation covariance
    Eigen::MatrixXd S = H * particle.covariance * H.transpose() + measurement.covariance_;
    
    // Compute Kalman gain
    Eigen::MatrixXd K = particle.covariance * H.transpose() * S.inverse();
    
    // Update mean
    updated_particle.mean = particle.mean + K * innovation;
    
    // Update covariance using Joseph form for stability
    // P = (I - KH)P(I - KH)' + KRK'
    int state_dim = particle.mean.size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd IKH = I - K * H;
    updated_particle.covariance = IKH * particle.covariance * IKH.transpose() + 
                                 K * measurement.covariance_ * K.transpose();
    
    // Compute likelihood for the association
    double likelihood = calculate_likelihood(particle, measurement);
    
    return {updated_particle, likelihood};
}
