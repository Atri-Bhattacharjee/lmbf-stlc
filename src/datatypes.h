#pragma once


/**
 * @file datatypes.h
 * @brief Core data structures for space debris tracking filter
 * 
 * This header defines the fundamental data structures for the space debris tracker.
 * These structures prioritize data locality and performance for the core C++ 
 * computational engine and are designed to be exposed to Python via pybind11.
 */

#include <vector>
#include <cstdint>
#include <string>
#include <Eigen/Dense>
#include <iostream>

/**
 * @brief A simple POD structure for creating unique, persistent track identities
 * 
 * This structure provides a unique identifier for each track, combining
 * temporal and sequential information for complete uniqueness.
 */
struct TrackLabel {
    uint64_t birth_time;  //!< The simulation time or epoch when the track was created
    uint32_t index;       //!< A unique index assigned at the time of birth
};

/**
 * @brief A POD structure representing a single, weighted state hypothesis
 * 
 * Each particle represents one possible state of the tracked object,
 * with an associated probability weight.
 */
struct Particle {
    //! [x, y, z, vx, vy, vz] where position is ECI in m, velocity is ECI in m/s
    Eigen::VectorXd state_vector;
    double weight;  //!< The probability weight of this particle
    Particle() : state_vector(6), weight(0.0) {}
};

/**
 * @brief Represents a single tracked object, its identity, and state uncertainty distribution
 * 
 * A track contains the unique identifier for the object and a cloud of weighted 
 * particles representing the probability density function of the object's state.
 */
class Track {
private:
    TrackLabel label_;                     //!< The unique, persistent label for this track
    double existence_probability_;         //!< The probability r that this track corresponds to a real object
    std::vector<Particle> particles_;     //!< The cloud of weighted particles representing the state probability density p(x)

public:
    /**
     * @brief Default constructor: label birth_time=0, index=0; existence_probability=0; empty particles
     */
    Track() : label_{0, 0}, existence_probability_(0.0), particles_{} {}

    /**
     * @brief Construct a new Track object
     * 
     * @param label The unique track label
     * @param existence_probability Initial existence probability
     * @param particles Initial particle cloud
     */
    Track(const TrackLabel& label, double existence_probability, const std::vector<Particle>& particles)
        : label_(label), existence_probability_(existence_probability), particles_(particles) {}

    /**
     * @brief Get the track label
     * @return const TrackLabel& Reference to the track label
     */
    const TrackLabel& label() const { return label_; }

    /**
     * @brief Get the existence probability
     * @return double The existence probability
     */
    double existence_probability() const { return existence_probability_; }

    /**
     * @brief Get the particles
     * @return const std::vector<Particle>& Reference to the particle vector
     */
    const std::vector<Particle>& particles() const { return particles_; }

    /**
     * @brief Set the existence probability
     * @param probability New existence probability
     */
    void set_existence_probability(double probability) { existence_probability_ = probability; }

    /**
     * @brief Set the particles
     * @param particles New particle cloud
     */
    void set_particles(const std::vector<Particle>& particles) { particles_ = particles; }
};

/**
 * @brief A POD structure for a single sensor detection
 * 
 * Contains all information about a measurement from a sensor,
 * including the measurement values, uncertainty, and source.
 */
struct Measurement {
    double timestamp_;                    //!< The epoch timestamp of the measurement
    Eigen::VectorXd value_;              //!< The 6D measurement vector [x, y, z, vx, vy, vz]
    Eigen::MatrixXd covariance_;         //!< The 6x6 measurement noise covariance matrix
    std::string sensor_id_;              //!< Identifier for the sensor that produced the measurement
    Eigen::VectorXd sensor_state_;           //!< The 6D ECI state of the sensor satellite

    Measurement()
        : timestamp_(0.0), value_(Eigen::VectorXd::Zero(6)), covariance_(Eigen::MatrixXd::Zero(6,6)), sensor_id_(), sensor_state_(Eigen::VectorXd::Zero(6)) {}
};

/**
 * @brief A container for the complete state of the LMB filter at a single point in time
 * 
 * This class holds the complete filter state, including all active tracks
 * and the timestamp of the current state.
 */
class FilterState {
private:
    double timestamp_;              //!< The timestamp of this filter state
    std::vector<Track> tracks_;     //!< The list of all current tracks

public:
    /**
     * @brief Default constructor: timestamp=0, empty tracks
     */
    FilterState() : timestamp_(0.0), tracks_{} {}

    FilterState(double timestamp, const std::vector<Track>& tracks)
        : timestamp_(timestamp), tracks_(tracks) {}

    /**
     * @brief Get the timestamp
     * @return double The filter state timestamp
     */
    double timestamp() const { return timestamp_; }

    // Non-const getter for direct modification
    std::vector<Track>& tracks() { return tracks_; }

    /**
     * @brief Get the tracks
     * @return const std::vector<Track>& Reference to the tracks vector
     */
    const std::vector<Track>& tracks() const { return tracks_; }

    /**
     * @brief Set the timestamp
     * @param timestamp New timestamp
     */
    void set_timestamp(double timestamp) { timestamp_ = timestamp; }

    /**
     * @brief Set the tracks
     * @param tracks New tracks vector
     */
    void set_tracks(const std::vector<Track>& tracks) { tracks_ = tracks; }
};