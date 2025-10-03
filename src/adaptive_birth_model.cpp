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
    
    // Initialize random number generator
    std::mt19937 gen(std::random_device{}());
    
    // Create standard normal distribution
    std::normal_distribution<> dist(0.0, 1.0);
    
    // Pre-calculate Cholesky decomposition of the initial covariance matrix
    Eigen::LLT<Eigen::MatrixXd> llt(initial_covariance_);
    Eigen::MatrixXd L = llt.matrixL();
    
    // Loop through each unused measurement
    for (size_t measurement_idx = 0; measurement_idx < unused_measurements.size(); ++measurement_idx) {
        const auto& measurement = unused_measurements[measurement_idx];
        
        // Create a new Track object (we'll set its properties below)
        TrackLabel label;
        label.birth_time = static_cast<uint64_t>(current_time);
        label.index = static_cast<uint32_t>(measurement_idx);
        
        // Create empty particle vector for the new track
        std::vector<Particle> particles;
        particles.reserve(particles_per_track_);
        
    // Define the mean state vector
    // Use the full 6D measurement value as the mean state
    Eigen::VectorXd mean_state(6);
    mean_state = measurement.value_;
        
        // Generate particles
        for (int particle_idx = 0; particle_idx < particles_per_track_; ++particle_idx) {
            Particle particle;
            // Generate 6x1 vector of standard normal random numbers
            Eigen::VectorXd standard_normal_vector(6);
            for (int i = 0; i < 6; ++i) {
                standard_normal_vector(i) = dist(gen);
            }
            // Create the final sampled state vector using Cholesky decomposition
            // sampled_state = mean_state + L * standard_normal_vector
            Eigen::VectorXd sampled_state = mean_state + L * standard_normal_vector;
            // Assign the sampled state to the particle's state vector
            particle.state_vector = sampled_state;
            // Assign equal weight to each particle
            particle.weight = 1.0 / particles_per_track_;
            // Add the particle to the track's particle vector
            particles.push_back(particle);
        }
        
        // Create the Track with the label, existence probability, and particles
        Track track(label, initial_existence_probability_, particles);
        
        // Add the fully formed Track to the new_tracks vector
        new_tracks.push_back(track);
    }
    
    // Return the vector of new tracks
    return new_tracks;
}