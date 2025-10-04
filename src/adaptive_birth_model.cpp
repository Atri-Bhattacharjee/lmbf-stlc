#include "adaptive_birth_model.h"
#include <random>
#include <vector>
#include <iostream> // Include iostream for debug prints

AdaptiveBirthModel::AdaptiveBirthModel(int particles_per_track, 
                                     double initial_existence_probability, 
                                     const Eigen::MatrixXd& initial_covariance)
    : particles_per_track_(particles_per_track),
      initial_existence_probability_(initial_existence_probability),
      initial_covariance_(initial_covariance) {
}

std::vector<Track> AdaptiveBirthModel::generate_new_tracks(const std::vector<Measurement>& unused_measurements, double current_time) const {
    // Initialize empty vector for new tracks
    std::vector<Track> new_tracks;
    
    // Loop through each unused measurement
    for (size_t measurement_idx = 0; measurement_idx < unused_measurements.size(); ++measurement_idx) {
        const auto& measurement = unused_measurements[measurement_idx];
        
        // Create a new Track object (we'll set its properties below)
        TrackLabel label;
        label.birth_time = static_cast<uint64_t>(current_time);
        label.index = static_cast<uint32_t>(measurement_idx);
        
        // In the EKF implementation, we use a single Gaussian component instead of multiple particles
        std::vector<Particle> gaussian_components;
        
        // Define the mean state vector - use the full 6D measurement value as the mean state
        Eigen::VectorXd mean_state = measurement.value_;
        
        // Create a single Gaussian component
        Particle gaussian_component;
        gaussian_component.mean = mean_state;
        gaussian_component.covariance = initial_covariance_;
        
        // Add the Gaussian component to the track
        gaussian_components.push_back(gaussian_component);
        
        // For multi-component Gaussian mixtures, we could add more components here
        // In this simple implementation, we just use a single component
        
        // Create the Track with the label, existence probability, and Gaussian components
        Track track(label, initial_existence_probability_, gaussian_components);
        
        // Add the fully formed Track to the new_tracks vector
        new_tracks.push_back(track);
    }
    
    // Return the vector of new tracks
    return new_tracks;
}