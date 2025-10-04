#pragma once

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>
#include <tuple>

class InOrbitSensorModel : public ISensorModel {
public:
    InOrbitSensorModel() = default;
    
    /**
     * @brief Calculate the likelihood of a measurement given a particle state
     * 
     * @param particle The particle representing a Gaussian state distribution
     * @param measurement The sensor measurement with covariance
     * @return double The likelihood value (probability density)
     */
    double calculate_likelihood(const Particle& particle, const Measurement& measurement) const override;
    
    /**
     * @brief Predicts the expected measurement from a given state
     * 
     * @param state The state vector to predict measurement from
     * @return Eigen::VectorXd The predicted measurement
     */
    Eigen::VectorXd predict_measurement(const Eigen::VectorXd& state) const override;
    
    /**
     * @brief Computes the Jacobian of the measurement model at a given state
     * 
     * @param state The state vector at which to compute the Jacobian
     * @return Eigen::MatrixXd The measurement model Jacobian matrix (H_k)
     */
    Eigen::MatrixXd get_measurement_jacobian(const Eigen::VectorXd& state) const override;
    
    /**
     * @brief Performs a complete EKF update of a particle with a measurement
     * 
     * @param particle The particle to update
     * @param measurement The measurement to use for the update
     * @return std::tuple<Particle, double> The updated particle and the likelihood of the association
     */
    std::tuple<Particle, double> ekf_update(const Particle& particle, const Measurement& measurement) const override;

private:
    // The Jacobian step size for finite differencing
    const double h_ = 1e-6;
};
