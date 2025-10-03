#include "linear_propagator.h"

Particle LinearPropagator::propagate(const Particle& particle, double dt, double current_time) const {
    Particle propagated_particle;
    propagated_particle.weight = particle.weight;
    
    Eigen::VectorXd new_state = particle.state_vector;
    // Simple linear propagation: x_new = x + v*dt
    new_state.head(3) += new_state.segment(3, 3) * dt;
    
    propagated_particle.state_vector = new_state;
    
    return propagated_particle;
}