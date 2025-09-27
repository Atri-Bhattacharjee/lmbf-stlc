#pragma once

/**
 * @file simple_sensor_model.h
 * @brief A basic sensor model for testing
 * 
 * This sensor model assumes both particle state and measurements are in 3D 
 * Cartesian coordinates and calculates likelihood based on Euclidean distance.
 * It is intended for the initial validation phase of the project and does not 
 * need to be physically accurate.
 */

#include "models.h"
#include "datatypes.h"

/**
 * @brief A basic sensor model for testing. It assumes both particle state and 
 * measurements are in 3D Cartesian coordinates and calculates likelihood based 
 * on Euclidean distance.
 */
class SimpleSensorModel : public ISensorModel {
public:
    /**
     * @brief Default constructor
     */
    SimpleSensorModel() = default;

    /**
     * @brief Default destructor
     */
    ~SimpleSensorModel() override = default;

    /**
     * @brief Calculate the likelihood of a measurement given a particle state
     * 
     * This method computes the probability density of observing the given
     * measurement if the true object state matches the provided particle.
     * This is a core component of the particle filter update step.
     * 
     * @param particle The particle representing a possible object state
     * @param measurement The sensor measurement with covariance
     * @return double The likelihood value (probability density)
     */
    double calculate_likelihood(const Particle& particle, const Measurement& measurement) const override;
};