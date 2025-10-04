#pragma once

/**
 * @file models.h
 * @brief Abstract interfaces for pluggable physics models in the high-performance simulation engine
 * 
 * This header defines the abstract base classes that serve as contracts for various
 * physics models including orbit propagation, sensor modeling, and track birth modeling.
 * These interfaces enable polymorphism and pluggable architecture for different
 * implementation strategies.
 */

#include "datatypes.h"
#include <vector>

/**
 * @brief Abstract base class that defines the interface for all orbit propagation models
 * 
 * This interface provides a contract for any orbit propagation implementation,
 * allowing different propagation algorithms (e.g., Two-Body, SGP4, high-fidelity)
 * to be used interchangeably within the simulation engine.
 */
class IOrbitPropagator {
public:
    /**
     * @brief Virtual destructor to enable proper polymorphic cleanup
     */
    virtual ~IOrbitPropagator() = default;

    /**
     * @brief Propagate a particle's state forward in time
     * 
     * This method takes a particle with a current state and propagates it
     * forward by the specified time interval using the implemented propagation model.
     * 
     * @param particle The input particle with current mean and covariance
     * @param dt The time step in seconds to propagate forward
     * @param process_noise The process noise covariance matrix Q
     * @return Particle The propagated particle with updated mean and covariance
     */
    virtual Particle propagate(const Particle& particle, double dt, const Eigen::MatrixXd& process_noise) const = 0;
    
    /**
     * @brief Computes the Jacobian of the motion model at a given state
     * 
     * This method calculates the state transition matrix (Jacobian) for the EKF
     * prediction step at the specified state point.
     * 
     * @param state The state vector at which to compute the Jacobian
     * @param dt The time step in seconds 
     * @return Eigen::MatrixXd The motion model Jacobian matrix (F_k)
     */
    virtual Eigen::MatrixXd get_motion_jacobian(const Eigen::VectorXd& state, double dt) const = 0;
};

/**
 * @brief Abstract base class that defines the interface for all sensor models
 * 
 * This interface provides a contract for sensor modeling implementations,
 * allowing different sensor types (e.g., radar, optical, RF) to calculate
 * measurement likelihoods in a consistent manner.
 */
class ISensorModel {
public:
    /**
     * @brief Virtual destructor to enable proper polymorphic cleanup
     */
    virtual ~ISensorModel() = default;

    /**
     * @brief Calculate the likelihood of a measurement given a particle state
     * 
     * This method computes the probability density of observing the given
     * measurement if the true object state matches the provided particle.
     * This is a core component of the filter update step.
     * 
     * @param particle The particle representing a Gaussian state distribution
     * @param measurement The sensor measurement with covariance
     * @return double The likelihood value (probability density)
     */
    virtual double calculate_likelihood(const Particle& particle, const Measurement& measurement) const = 0;
    
    /**
     * @brief Predicts the expected measurement from a given state
     * 
     * This method applies the measurement function h(x) to predict what
     * measurement would result from a given state.
     * 
     * @param state The state vector to predict measurement from
     * @return Eigen::VectorXd The predicted measurement
     */
    virtual Eigen::VectorXd predict_measurement(const Eigen::VectorXd& state) const = 0;
    
    /**
     * @brief Computes the Jacobian of the measurement model at a given state
     * 
     * This method calculates the measurement model Jacobian for the EKF
     * update step at the specified state point.
     * 
     * @param state The state vector at which to compute the Jacobian
     * @return Eigen::MatrixXd The measurement model Jacobian matrix (H_k)
     */
    virtual Eigen::MatrixXd get_measurement_jacobian(const Eigen::VectorXd& state) const = 0;
    
    /**
     * @brief Performs a complete EKF update of a particle with a measurement
     * 
     * This method combines the measurement prediction, Jacobian calculation, and
     * the EKF update equations to produce an updated particle and likelihood.
     * 
     * @param particle The particle to update
     * @param measurement The measurement to use for the update
     * @return std::tuple<Particle, double> The updated particle and the likelihood of the association
     */
    virtual std::tuple<Particle, double> ekf_update(const Particle& particle, const Measurement& measurement) const = 0;
};

/**
 * @brief Abstract base class that defines the interface for track birth models
 * 
 * This interface provides a contract for track initialization strategies,
 * allowing different approaches for creating new tracks from unused measurements
 * (e.g., single-measurement birth, multiple-measurement confirmation).
 */
class IBirthModel {
public:
    /**
     * @brief Virtual destructor to enable proper polymorphic cleanup
     */
    virtual ~IBirthModel() = default;

    /**
     * @brief Generate new tracks from unused measurements
     * 
     * This method analyzes measurements that were not associated with existing
     * tracks and creates new track hypotheses. The implementation determines
     * the birth strategy and initial particle distributions.
     * 
     * @param unused_measurements Vector of measurements not associated with existing tracks
     * @param current_time The current simulation time for track initialization
     * @return std::vector<Track> Vector of newly created tracks with initial particle clouds
     */
    virtual std::vector<Track> generate_new_tracks(const std::vector<Measurement>& unused_measurements, double current_time) const = 0;
};