#pragma once
#include <vector>
#include "datatypes.h"
#include <Eigen/Dense>

double calculate_ospa_distance(const std::vector<Track>& tracks,
                               const std::vector<Eigen::VectorXd>& ground_truths,
                               double cutoff);
