#pragma once

#include <vector>
#include <cstdint>

// Core data structures for the LMB space debris tracking system
// This header will contain the main data types used throughout the project

/**
 * @brief Represents a particle in the particle filter
 * 
 * A particle contains the state information and weight for a single hypothesis
 * in the particle filter representation of a track.
 */
struct Particle {
    std::vector<double> state;  ///< State vector (position, velocity, etc.)
    double weight = 1.0;        ///< Particle weight
    
    Particle() = default;
    Particle(const std::vector<double>& s, double w = 1.0) : state(s), weight(w) {}
};

/**
 * @brief Represents a measurement from a sensor
 * 
 * Contains the observed data from a sensor at a specific time,
 * along with associated uncertainty information.
 */
struct Measurement {
    std::vector<double> data;   ///< Measurement vector
    double timestamp = 0.0;     ///< Time of measurement
    std::uint32_t sensor_id = 0; ///< ID of the sensor that made this measurement
    
    Measurement() = default;
    Measurement(const std::vector<double>& d, double t = 0.0, std::uint32_t id = 0) 
        : data(d), timestamp(t), sensor_id(id) {}
};

/**
 * @brief Represents a track in the multi-target tracking system
 * 
 * A track contains a collection of particles representing the probability
 * distribution of a target's state, along with track management information.
 */
struct Track {
    std::vector<Particle> particles; ///< Particle representation of the track
    double birth_time = 0.0;         ///< Time when track was initiated
    std::uint64_t track_id = 0;      ///< Unique track identifier
    double existence_probability = 0.0; ///< Probability that this track represents a real target
    
    Track() = default;
    Track(const std::vector<Particle>& p, double bt = 0.0, std::uint64_t id = 0, double ep = 0.0)
        : particles(p), birth_time(bt), track_id(id), existence_probability(ep) {}
};
