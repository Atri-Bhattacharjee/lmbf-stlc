#include "sgp4_propagator.h"
#include "SGP4.h"
#include "Tle.h"
#include "Eci.h"
#include "DateTime.h"
#include <random>
#include <string>
#include <Eigen/Dense>

SGP4Propagator::SGP4Propagator(const Eigen::MatrixXd& process_noise_covariance)
    : process_noise_covariance_(process_noise_covariance) {}

Particle SGP4Propagator::propagate(const Particle& particle, double dt) const {
    Particle propagated_particle = particle;
    // Use types from libsgp4 namespace
    libsgp4::Tle tle("OBJECT_NAME", particle.tle_line1, particle.tle_line2);
    libsgp4::SGP4 sgp4(tle);
    libsgp4::DateTime epoch = tle.Epoch();
    libsgp4::DateTime future_dt = epoch.AddSeconds(dt);
    libsgp4::Eci eci_state = sgp4.FindPosition(future_dt);

    Eigen::VectorXd propagated_state(6);
    propagated_state(0) = eci_state.Position().x;
    propagated_state(1) = eci_state.Position().y;
    propagated_state(2) = eci_state.Position().z;
    propagated_state(3) = eci_state.Velocity().x;
    propagated_state(4) = eci_state.Velocity().y;
    propagated_state(5) = eci_state.Velocity().z;

    // Add process noise
    static std::random_device rd;
    static std::mt19937 gen(rd());
    Eigen::LLT<Eigen::MatrixXd> llt(process_noise_covariance_);
    Eigen::MatrixXd L = llt.matrixL();
    Eigen::VectorXd noise(6);
    for (int i = 0; i < 6; ++i) {
        std::normal_distribution<> d(0.0, 1.0);
        noise(i) = d(gen);
    }
    propagated_state += L * noise;

    propagated_particle.cartesian_state_vector = propagated_state;
    return propagated_particle;
}
