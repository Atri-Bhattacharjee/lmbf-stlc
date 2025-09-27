#pragma once

/**
 * @file smc_lmb_tracker.h
 * @brief Central class for Sequential Monte Carlo Labeled Multi-Bernoulli tracking filter
 * 
 * This header defines the SMC_LMB_Tracker class which orchestrates the entire 
 * filtering process using pluggable model interfaces for orbit propagation,
 * sensor modeling, and track birth modeling.
 */

#include <vector>
#include <memory>
#include "datatypes.h"
#include "models.h"

/**
 * @brief Sequential Monte Carlo Labeled Multi-Bernoulli Tracker
 * 
 * This class implements the central orchestration logic for a Bayesian tracking
 * filter using the Labeled Multi-Bernoulli framework. It coordinates between
 * different pluggable model components to perform prediction and update steps
 * for multiple target tracking.
 */
class SMC_LMB_Tracker {
private:
    FilterState current_state_;                         //!< Stores the current list of tracks and timestamp
    std::unique_ptr<IOrbitPropagator> propagator_;     //!< Smart pointer to the propagator model
    std::unique_ptr<ISensorModel> sensor_model_;       //!< Smart pointer to the sensor model
    std::unique_ptr<IBirthModel> birth_model_;         //!< Smart pointer to the birth model
    double survival_probability_;                       //!< Configuration parameter for track survival probability

public:
    /**
     * @brief Construct a new SMC_LMB_Tracker object
     * 
     * @param propagator Unique pointer to orbit propagation model
     * @param sensor_model Unique pointer to sensor model
     * @param birth_model Unique pointer to birth model
     * @param survival_probability Probability of a track surviving a time step
     */
    SMC_LMB_Tracker(std::unique_ptr<IOrbitPropagator> propagator,
                    std::unique_ptr<ISensorModel> sensor_model,
                    std::unique_ptr<IBirthModel> birth_model,
                    double survival_probability);

    /**
     * @brief Run the predict step of the filter
     * 
     * This method propagates all existing tracks forward in time using the
     * configured orbit propagator model.
     * 
     * @param dt Time step in seconds to propagate forward
     */
    void predict(double dt);

    /**
     * @brief Run the update step of the filter
     * 
     * This method updates track existence probabilities and particle weights
     * based on new sensor measurements, and creates new tracks from unused
     * measurements.
     * 
     * @param measurements Vector of new sensor measurements to process
     */
    void update(const std::vector<Measurement>& measurements);

    /**
     * @brief Get the current tracks
     * 
     * Returns a const reference to the current track list for efficiency.
     * 
     * @return const std::vector<Track>& Reference to the current tracks
     */
    const std::vector<Track>& get_tracks() const;

    /**
     * @brief Set the tracks for testing purposes
     * 
     * This method allows initialization of the filter's state for testing
     * and debugging purposes.
     * 
     * @param tracks Vector of tracks to set as the current filter state
     */
    void set_tracks(const std::vector<Track>& tracks);
};