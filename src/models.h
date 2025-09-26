#pragma once

#include "datatypes.h"
#include <vector>

/**
 * @file models.h
 * @brief Abstract interfaces for pluggable physics models in the high-performance simulation engine
 * 
 * This header defines the abstract base classes that serve as contracts for various
 * concrete implementations of physics models used in the LMB space debris tracking system.
 * All interfaces are designed to be thread-safe and const-correct.
 */

/**
 * @class IOrbitPropagator
 * @brief Abstract base class that defines the interface for all orbit propagation models
 * 
 * Orbit propagators are responsible for advancing the state of a particle (representing
 * a space object) forward in time according to the underlying physics model (e.g., 
 * two-body dynamics, J2 perturbations, atmospheric drag, etc.).
 */
class IOrbitPropagator {
public:
    /**
     * @brief Virtual destructor for proper polymorphic destruction
     */
    virtual ~IOrbitPropagator() = default;

    /**
     * @brief Propagates a particle's state forward in time
     * 
     * @param particle The input particle containing the current state
     * @param dt Time step for propagation (in seconds)
     * @return New particle with propagated state
     * 
     * @note This method is const as propagators should not modify their internal state
     *       during propagation operations.
     */
    virtual Particle propagate(const Particle& particle, double dt) const = 0;
};

/**
 * @class ISensorModel
 * @brief Abstract base class that defines the interface for all sensor models
 * 
 * Sensor models compute the likelihood of a measurement given a particle's predicted
 * state. This is a key component in particle filtering and data association algorithms.
 */
class ISensorModel {
public:
    /**
     * @brief Virtual destructor for proper polymorphic destruction
     */
    virtual ~ISensorModel() = default;

    /**
     * @brief Calculates the likelihood of a measurement given a particle's state
     * 
     * @param particle The particle containing the predicted state
     * @param measurement The observed measurement from a sensor
     * @return Likelihood value (probability density or log-likelihood)
     * 
     * @note This method is const as sensor models should not modify their internal state
     *       during likelihood calculations.
     */
    virtual double calculate_likelihood(const Particle& particle, const Measurement& measurement) const = 0;
};

/**
 * @class IBirthModel
 * @brief Abstract base class that defines the interface for track birth models
 * 
 * Birth models are responsible for generating new track hypotheses from measurements
 * that are not associated with existing tracks. This is essential for detecting
 * new targets entering the surveillance region.
 */
class IBirthModel {
public:
    /**
     * @brief Virtual destructor for proper polymorphic destruction
     */
    virtual ~IBirthModel() = default;

    /**
     * @brief Generates new track hypotheses from unused measurements
     * 
     * @param unused_measurements Vector of measurements not associated with existing tracks
     * @param current_time Current simulation time for track initialization
     * @return Vector of new tracks generated from the unused measurements
     * 
     * @note This method is const as birth models should not modify their internal state
     *       during track generation.
     */
    virtual std::vector<Track> generate_new_tracks(const std::vector<Measurement>& unused_measurements, 
                                                   double current_time) const = 0;
};