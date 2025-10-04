#include "two_body_propagator.h"
#include <Eigen/Dense>
#include <random>
#include <cmath>

// Helper function: Compute state derivative for two-body problem
Eigen::VectorXd calculate_state_derivative(const Eigen::VectorXd& state_6d) {
    constexpr double mu = 3.986004418e14; // Earth's gravitational parameter (m^3/s^2)
    Eigen::Vector3d pos = state_6d.head(3);
    Eigen::Vector3d vel = state_6d.segment(3, 3);
    double r_norm = pos.norm();
    Eigen::Vector3d acc = -mu * pos / std::pow(r_norm, 3);
    Eigen::VectorXd dydt(6);
    dydt.head(3) = vel;
    dydt.tail(3) = acc;
    return dydt;
}

Eigen::VectorXd TwoBodyPropagator::propagate_state(const Eigen::VectorXd& state, double dt) const {
    // RK4 integration
    Eigen::VectorXd k1 = calculate_state_derivative(state);
    Eigen::VectorXd k2 = calculate_state_derivative(state + 0.5 * dt * k1);
    Eigen::VectorXd k3 = calculate_state_derivative(state + 0.5 * dt * k2);
    Eigen::VectorXd k4 = calculate_state_derivative(state + dt * k3);
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

Particle TwoBodyPropagator::propagate(const Particle& particle, double dt, const Eigen::MatrixXd& process_noise) const {
    // Extract initial state
    Eigen::VectorXd mean_predicted = propagate_state(particle.mean, dt);
    
    // Calculate state transition Jacobian
    Eigen::MatrixXd F = get_motion_jacobian(particle.mean, dt);
    
    // EKF covariance prediction
    Eigen::MatrixXd covariance_predicted = F * particle.covariance * F.transpose() + process_noise;
    
    // Create propagated particle
    Particle propagated_particle;
    propagated_particle.mean = mean_predicted;
    propagated_particle.covariance = covariance_predicted;
    
    return propagated_particle;
}

Eigen::MatrixXd TwoBodyPropagator::get_motion_jacobian(const Eigen::VectorXd& state, double dt) const {
    const double epsilon = 1e-6; // Small perturbation for finite differencing
    Eigen::MatrixXd jacobian(6, 6);
    
    // Compute the nominal propagated state
    Eigen::VectorXd nominal_result = propagate_state(state, dt);
    
    // Compute the Jacobian using central finite differences
    for (int i = 0; i < 6; ++i) {
        // Create perturbed states
        Eigen::VectorXd state_plus = state;
        Eigen::VectorXd state_minus = state;
        
        state_plus(i) += epsilon;
        state_minus(i) -= epsilon;
        
        // Propagate perturbed states
        Eigen::VectorXd result_plus = propagate_state(state_plus, dt);
        Eigen::VectorXd result_minus = propagate_state(state_minus, dt);
        
        // Compute the column of the Jacobian using central difference
        jacobian.col(i) = (result_plus - result_minus) / (2.0 * epsilon);
    }
    
    return jacobian;
}
