#pragma once

/**
 * @file linear_propagator.h
 * @brief A simple linear propagator implementation for orbit propagation
 */

#include "models.h"
#include "datatypes.h"

/**
 * @brief A simple, non-physical propagator that moves an object in a straight line at a constant velocity. Useful for testing.
 * 
 * This propagator implements the IOrbitPropagator interface and provides a basic
 * linear motion model where objects continue moving at constant velocity without
 * any forces applied. This is primarily useful for testing and debugging purposes.
 */
class LinearPropagator : public IOrbitPropagator {
public:
    /**
     * @brief Default constructor
     */
    LinearPropagator() = default;

    /**
     * @brief Default destructor
     */
    ~LinearPropagator() override = default;

    /**
     * @brief Propagate a particle's state forward in time using linear motion
     * @param dt The time step in seconds to propagate forward
     * @param current_time The current simulation timestamp (unused)
     * @return Particle The propagated particle with updated state
     */
    Particle propagate(const Particle& particle, double dt, double current_time) const override;
};