#pragma once

/**
 * @file two_body_propagator.h
 * @brief Two-Body orbital propagation model implementation
 * 
 * This header defines the TwoBodyPropagator class which implements
 * the IOrbitPropagator interface using two-body orbital mechanics with EKF.
 */

#include "models.h"
#include <Eigen/Dense>

/**
 * @brief Two-Body orbital propagator implementation with EKF support
 * 
 * This class provides a physics-based orbital propagation model using
 * two-body (Keplerian) orbital mechanics, adapted for use with EKF.
 */
class TwoBodyPropagator : public IOrbitPropagator {
public:
    /**
     * @brief Default constructor
     */
    TwoBodyPropagator() {}

    /**
     * @brief Propagate a particle's state using two-body orbital mechanics and EKF
     * 
     * This method implements the IOrbitPropagator interface, propagating
     * both the mean and covariance of a Gaussian state distribution.
     * 
     * @param particle The input particle to propagate
     * @param dt The time step in seconds
     * @param process_noise The process noise covariance matrix
     * @return Particle The propagated particle with updated mean and covariance
     */
    Particle propagate(const Particle& particle, double dt, const Eigen::MatrixXd& process_noise) const override;

    /**
     * @brief Computes the Jacobian of the two-body motion model
     * 
     * This method calculates the state transition matrix for the two-body
     * propagation model using numerical finite differencing.
     * 
     * @param state The state vector at which to compute the Jacobian
     * @param dt The time step in seconds
     * @return Eigen::MatrixXd The 6x6 motion model Jacobian matrix
     */
    Eigen::MatrixXd get_motion_jacobian(const Eigen::VectorXd& state, double dt) const override;

private:
    /**
     * @brief Helper function to propagate a state vector using RK4 integration
     * 
     * @param state The initial state vector
     * @param dt The time step in seconds
     * @return Eigen::VectorXd The propagated state vector
     */
    Eigen::VectorXd propagate_state(const Eigen::VectorXd& state, double dt) const;
};
