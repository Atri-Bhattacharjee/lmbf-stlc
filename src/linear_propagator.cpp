#include "linear_propagator.h"

Particle LinearPropagator::propagate(const Particle& particle, double dt) const {
    Particle propagated_particle;
    propagated_particle.weight = particle.weight;
    
    Eigen::VectorXd new_state = particle.state_vector;
    auto pos = new_state.head<3>();
    auto vel = new_state.segment<3>(3);
    
    pos += vel * dt;
    
    propagated_particle.state_vector = new_state;
    
    return propagated_particle;
}