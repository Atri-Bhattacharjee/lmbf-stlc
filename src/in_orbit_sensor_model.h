#pragma once

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>

class InOrbitSensorModel : public ISensorModel {
public:
    InOrbitSensorModel() = default;
    double calculate_likelihood(const Particle& particle, const Measurement& measurement) const override;
};
