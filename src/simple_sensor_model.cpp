#include "simple_sensor_model.h"
#include <cmath>

double SimpleSensorModel::calculate_likelihood(const Particle& particle, const Measurement& measurement) const {
    // Extract the 3D position vector from the particle's state (first 3 elements)
    Eigen::Vector3d particle_position = particle.state_vector.head<3>();
    
    // The measurement.value_ is assumed to be a 3D Cartesian position vector
    // Calculate the residual (difference between measurement and particle position)
    Eigen::Vector3d residual = measurement.value_.head<3>() - particle_position;
    
    // Calculate the Mahalanobis distance squared
    // For this simple test model, we assume a diagonal covariance
    // Get the inverse of the measurement covariance matrix
    Eigen::Matrix3d covariance_inv = measurement.covariance_.inverse();
    
    // Calculate the Mahalanobis distance squared
    double mahalanobis_sq = residual.transpose() * covariance_inv * residual;
    
    // Calculate the final likelihood score using the formula for a multivariate Gaussian PDF
    // We ignore the normalization constant (which is not needed for the filter's relative weighting)
    // The formula is exp(-0.5 * mahalanobis_sq)
    return std::exp(-0.5 * mahalanobis_sq);
}