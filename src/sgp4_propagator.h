#pragma once

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>

class SGP4Propagator : public IOrbitPropagator {
private:
    Eigen::MatrixXd process_noise_covariance_; // 7x7
public:
    SGP4Propagator(const Eigen::MatrixXd& process_noise_covariance);
    Particle propagate(const Particle& particle, double dt, double current_time) const override;
};
