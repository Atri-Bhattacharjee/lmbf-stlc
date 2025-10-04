#include "smc_lmb_tracker.h"
#include "assignment.h"
#include <iostream>

SMC_LMB_Tracker::SMC_LMB_Tracker(std::shared_ptr<IOrbitPropagator> propagator,
                                                                 std::shared_ptr<ISensorModel> sensor_model,
                                                                 std::shared_ptr<IBirthModel> birth_model,
                                                                 double survival_probability)
        : current_state_(0.0, std::vector<Track>{}),
            propagator_(std::move(propagator)),
            sensor_model_(std::move(sensor_model)),
            birth_model_(std::move(birth_model)),
            survival_probability_(survival_probability) {
}

void SMC_LMB_Tracker::predict(double dt) {
    // Projects the filter state forward in time by operating in-place.
    const double previous_time = current_state_.timestamp();
    current_state_.set_timestamp(current_state_.timestamp() + dt);
    
    // Process noise covariance matrix Q for the EKF propagation
    // This is a simple diagonal matrix with position and velocity noise
    // In a real implementation, this would be tuned based on system characteristics
    Eigen::MatrixXd process_noise = Eigen::MatrixXd::Zero(6, 6);
    process_noise.diagonal() << 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4; // Position (m^2), Velocity (m^2/s^2)
    
    // Get a direct, MODIFIABLE reference to the internal track vector.
    std::vector<Track>& tracks = current_state_.tracks();
    for (Track& track : tracks) {
        // Update existence probability in-place.
        track.set_existence_probability(track.existence_probability() * survival_probability_);
        
        // In the EKF version, we propagate each Gaussian component
        std::vector<Particle> propagated_particles;
        propagated_particles.reserve(track.particles().size());
        
        for (const Particle& particle : track.particles()) {
            // Use the IOrbitPropagator interface's propagate method which now handles
            // EKF prediction with state and covariance
            Particle new_particle = propagator_->propagate(particle, dt, process_noise);
            propagated_particles.push_back(new_particle);
        }
        
        // Replace the track's old particle cloud with the new one.
        track.set_particles(propagated_particles);
    }
    // No need to call set_tracks(), as we have modified the state directly.
}

void SMC_LMB_Tracker::update(const std::vector<Measurement>& measurements) {
    std::vector<Track>& tracks = current_state_.tracks();
    size_t num_tracks = tracks.size();
    size_t num_meas = measurements.size();

    // Handle the case of no existing tracks - just create new ones from all measurements
    if (num_tracks == 0) {
        std::vector<Track> born_tracks = birth_model_->generate_new_tracks(measurements, current_state_.timestamp());
        current_state_.tracks().swap(born_tracks);
        return;
    }

    // Step 1: Build a cost matrix for tracks and measurements
    // For the EKF implementation, we need to handle potentially rectangular case
    // by adding dummy rows or columns to the cost matrix
    
    // Determine the size of the cost matrix (max of tracks and measurements)
    size_t cost_matrix_size = std::max(num_tracks, num_meas);
    Eigen::MatrixXd cost_matrix = Eigen::MatrixXd::Constant(
        cost_matrix_size, cost_matrix_size, 1000.0); // High cost for dummy assignments
        
    // Pre-compute all EKF updates and likelihoods
    std::vector<std::vector<std::tuple<Particle, double>>> updated_particles_and_likelihoods(num_tracks);
    
    // For each track-measurement pair, compute the EKF update and likelihood
    for (size_t i = 0; i < num_tracks; ++i) {
        updated_particles_and_likelihoods[i].resize(num_meas);
        
        for (size_t j = 0; j < num_meas; ++j) {
            // For each Gaussian component in the track (typically just one in EKF)
            const auto& current_track = tracks[i];
            const auto& current_particle = current_track.particles()[0]; // Assume single Gaussian
            
            // Perform EKF update and get likelihood using the sensor model
            std::tuple<Particle, double> update_result = 
                sensor_model_->ekf_update(current_particle, measurements[j]);
            
            // Store the updated particle and likelihood
            updated_particles_and_likelihoods[i][j] = update_result;
            
            // Store the negative log-likelihood as the cost
            double likelihood = std::get<1>(update_result);
            double track_r = current_track.existence_probability();
            cost_matrix(i, j) = -std::log(std::max(track_r * likelihood, 1e-12));
        }
    }
    
    // --- INSERT NEW GATING STEP ---
    // Gating breaks ambiguity for symmetric targets by using the prediction as an anchor.
    const double gating_threshold_sq = 500.0 * 500.0; // A 500m gate (squared for efficiency)

    // For each track-measurement pair
    for (size_t i = 0; i < num_tracks; ++i) {
        // Get the predicted mean state from the original track (before any EKF updates)
        const Eigen::VectorXd& predicted_mean = tracks[i].particles()[0].mean;

        for (size_t j = 0; j < num_meas; ++j) {
            // Calculate the squared Euclidean distance in position
            double dist_sq = (predicted_mean.head(3) - measurements[j].value_.head(3)).squaredNorm();

            // If the measurement is outside the gate for this track...
            if (dist_sq > gating_threshold_sq) {
                // ...this association is physically impossible. Set its likelihood to zero.
                // In the negative log-likelihood domain, set to a very large number
                cost_matrix(i, j) = 1000.0; // Effectively infinity for the assignment problem
            }
        }
    }
    // --- END OF NEW GATING STEP ---
    
    // Step 2: Solve assignment (K-best)
    std::vector<Hypothesis> hypotheses = solve_assignment(cost_matrix, 100);
    
    // Normalize hypothesis weights (log-sum-exp)
    std::vector<double> log_weights;
    for (const auto& h : hypotheses) log_weights.push_back(-h.weight);
    double max_logw = *std::max_element(log_weights.begin(), log_weights.end());
    std::vector<double> norm_weights;
    double sum_exp = 0.0;
    for (double lw : log_weights) sum_exp += std::exp(lw - max_logw);
    for (double lw : log_weights) norm_weights.push_back(std::exp(lw - max_logw) / sum_exp);
    
    // Step 3: Create updated tracks based on best assignments
    std::vector<Track> updated_tracks;
    for (size_t i = 0; i < num_tracks; ++i) {
        // Find the best assignment for this track
        int best_assoc_idx = hypotheses[0].associations[i];
        
        // Only consider valid measurement associations
        if (best_assoc_idx >= 0 && best_assoc_idx < num_meas) {
            // Get the updated particle and likelihood for this assignment
            const auto& update_result = updated_particles_and_likelihoods[i][best_assoc_idx];
            const Particle& updated_particle = std::get<0>(update_result);
            double likelihood = std::get<1>(update_result);
            
            // Create a vector with just this one updated Gaussian component
            std::vector<Particle> updated_particles = {updated_particle};
            
            // Update the track's existence probability
            double track_r = tracks[i].existence_probability();
            double new_existence_probability = track_r * likelihood / 
                (track_r * likelihood + (1 - track_r));
            
            // Create the updated track
            Track updated_track = tracks[i];
            updated_track.set_existence_probability(new_existence_probability);
            updated_track.set_particles(updated_particles);
            updated_tracks.push_back(updated_track);
        }
        else {
            // No valid association - this is a missed detection
            // The track survives but with reduced existence probability
            Track missed_track = tracks[i];
            double missed_existence_prob = missed_track.existence_probability() * 0.9; // Decay factor
            missed_track.set_existence_probability(missed_existence_prob);
            updated_tracks.push_back(missed_track);
        }
    }
    
    // Step 4: Handle birth of new tracks from unassociated measurements
    std::vector<Measurement> unused_measurements;
    for (size_t j = 0; j < num_meas; ++j) {
        bool is_used = false;
        for (const auto& h : hypotheses) {
            for (int assoc : h.associations) {
                if (assoc == j) {
                    is_used = true;
                    break;
                }
            }
            if (is_used) break;
        }
        if (!is_used) {
            unused_measurements.push_back(measurements[j]);
        }
    }
    
    std::vector<Track> born_tracks = birth_model_->generate_new_tracks(
        unused_measurements, current_state_.timestamp());
    
    // Step 5: Track pruning - remove tracks with low existence probability
    const double prune_threshold = 0.01; // configurable
    std::vector<Track> pruned_tracks;
    for (const auto& track : updated_tracks) {
        if (track.existence_probability() >= prune_threshold) {
            pruned_tracks.push_back(track);
        }
    }
    
    // Add new born tracks
    pruned_tracks.insert(pruned_tracks.end(), born_tracks.begin(), born_tracks.end());
    
    // Update the current state with the new tracks
    current_state_.tracks().swap(pruned_tracks);
}

// Helper: average likelihood over all particles

double SMC_LMB_Tracker::compute_association_likelihood(const Track& track, const Measurement& measurement) const {
    // In the EKF version, each track has Gaussian components instead of particles with weights
    // We calculate the likelihood for each Gaussian component and combine them
    double total_likelihood = 0.0;
    const auto& particles = track.particles();
    
    // For each Gaussian component
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        // Calculate likelihood using the EKF likelihood calculation
        double component_likelihood = sensor_model_->calculate_likelihood(p, measurement);
        // In EKF-LMB, we don't have weights for Gaussian components within a track,
        // but we would average over multiple components if we had them
        total_likelihood += component_likelihood / particles.size();
    }
    
    return total_likelihood;
}

const std::vector<Track>& SMC_LMB_Tracker::get_tracks() const {
    return current_state_.tracks();
}

void SMC_LMB_Tracker::set_tracks(const std::vector<Track>& tracks) {
    current_state_.set_tracks(tracks);
}