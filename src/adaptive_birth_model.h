#pragma once

/**
 * @file adaptive_birth_model.h
 * @brief Implementation of adaptive birth model for track initialization
 * 
 * This class implements the IBirthModel interface to create new track hypotheses
 * from sensor measurements that were not associated with existing tracks. It uses
 * an adaptive approach with configurable parameters for particle generation.
 */

#include "models.h"
#include "datatypes.h"
#include <Eigen/Dense>

/**
 * @brief Adaptive birth model for creating new tracks from unused measurements
 * 
 * This class implements track birth by generating a configurable number of particles
 * for each unused measurement. The particles are sampled from a multivariate normal
 * distribution centered at the measurement position with configurable covariance.
 */
class AdaptiveBirthModel : public IBirthModel {
private:
    int particles_per_track_;                    //!< The number of particles to generate for each new track
    double initial_existence_probability_;       //!< The low probability assigned to a brand new track
    Eigen::MatrixXd initial_covariance_;         //!< A 7x7 matrix defining the initial uncertainty

public:
    /**
     * @brief Construct a new Adaptive Birth Model object
     * 
     * @param particles_per_track Number of particles to generate for each new track
     * @param initial_existence_probability The initial existence probability for new tracks
     * @param initial_covariance 7x7 covariance matrix for initial state uncertainty
     */
    AdaptiveBirthModel(int particles_per_track, 
                      double initial_existence_probability, 
                      const Eigen::MatrixXd& initial_covariance);

    /**
     * @brief Default destructor
     */
    ~AdaptiveBirthModel() override = default;

    /**
     * @brief Generate new tracks from unused measurements
     * 
     * Creates new track hypotheses by generating particle clouds for each unused
     * measurement. Each particle is sampled from a multivariate normal distribution
     * using Cholesky decomposition for efficient sampling.
     * 
     * @param unused_measurements Vector of measurements not associated with existing tracks
     * @param current_time The current simulation time for track initialization
     * @return std::vector<Track> Vector of newly created tracks with initial particle clouds
     */
    std::vector<Track> generate_new_tracks(const std::vector<Measurement>& unused_measurements, 
                                          double current_time) const override;
};