#include "smc_lmb_tracker.h"
#include "assignment.h"

SMC_LMB_Tracker::SMC_LMB_Tracker(std::shared_ptr<IOrbitPropagator> propagator,
                                                                 std::shared_ptr<ISensorModel> sensor_model,
                                                                 std::shared_ptr<IBirthModel> birth_model,
                                                                 double survival_probability,
                                                                 double detection_probability)
        : current_state_(0.0, std::vector<Track>{}),
            propagator_(std::move(propagator)),
            sensor_model_(std::move(sensor_model)),
            birth_model_(std::move(birth_model)),
            survival_probability_(survival_probability),
            detection_probability_(detection_probability) {
}

void SMC_LMB_Tracker::predict(double dt) {
    // Projects the filter state forward in time by operating in-place.
    const double previous_time = current_state_.timestamp();
    current_state_.set_timestamp(current_state_.timestamp() + dt);
    // Get a direct, MODIFIABLE reference to the internal track vector.
    std::vector<Track>& tracks = current_state_.tracks();
    for (Track& track : tracks) {
        // Update existence probability in-place.
        track.set_existence_probability(track.existence_probability() * survival_probability_);
        // Propagate particles.
        std::vector<Particle> propagated_particles;
        propagated_particles.reserve(track.particles().size());
        for (const Particle& particle : track.particles()) {
            Particle new_particle = propagator_->propagate(particle, dt, previous_time);
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

    // Step 1: Create all hypothetical updated particle sets
    // This 3D vector stores all hypothetical particle sets: [track_idx][hyp_idx][particle_idx]
    // Where hyp_idx: 0 to num_meas-1 are measurement associations, and hyp_idx=num_meas is missed detection
    std::vector<std::vector<std::vector<Particle>>> hypothetical_particle_sets(num_tracks);
    Eigen::MatrixXd likelihood_matrix(num_tracks, num_meas);

    // For each track
    for (size_t i = 0; i < num_tracks; ++i) {
        const auto& current_particles = tracks[i].particles();
        size_t num_particles = current_particles.size();

        // Initialize storage for this track's hypothetical sets
        hypothetical_particle_sets[i].resize(num_meas + 1); // +1 for missed detection case
        
        // For each measurement, create a hypothetically updated set
        for (size_t j = 0; j < num_meas; ++j) {
            const auto& measurement = measurements[j];
            hypothetical_particle_sets[i][j].reserve(num_particles);
            
            double total_likelihood = 0.0;

            // For each particle
            for (size_t p = 0; p < num_particles; ++p) {
                const auto& current_particle = current_particles[p];
                
               
                Particle updated_particle;
                
                
                
                // Calculate likelihood and update weight
                double particle_likelihood = sensor_model_->calculate_likelihood(current_particle, measurement);
                updated_particle.weight = current_particle.weight * particle_likelihood;
                total_likelihood += updated_particle.weight;
                
                // Add to hypothetical set
                hypothetical_particle_sets[i][j].push_back(updated_particle);
            }
            
            // Store the total likelihood for this track-measurement association
            likelihood_matrix(i, j) = total_likelihood;
            
            // Normalize weights within this hypothetical set
            if (total_likelihood > 1e-12) {
                for (auto& particle : hypothetical_particle_sets[i][j]) {
                    particle.weight /= total_likelihood;
                }
            } else {
                // Handle the case of zero likelihood
                for (auto& particle : hypothetical_particle_sets[i][j]) {
                    particle.weight = 1.0 / num_particles;
                }
            }
        }
        
        // Create the missed detection set (states unchanged, weights preserved)
        hypothetical_particle_sets[i][num_meas].reserve(num_particles);
        for (const auto& current_particle : current_particles) {
            Particle missed_particle;
            missed_particle.state_vector = current_particle.state_vector; // State unchanged
            missed_particle.weight = current_particle.weight; // Weight preserved
            hypothetical_particle_sets[i][num_meas].push_back(missed_particle);
        }
    }
    
    // Step 2: Build cost matrix using the likelihoods from hypothetical updates
    Eigen::MatrixXd cost_matrix(num_tracks, num_meas + num_tracks);
    for (size_t i = 0; i < num_tracks; ++i) {
        double track_r = tracks[i].existence_probability();
        
        // Association costs
        for (size_t j = 0; j < num_meas; ++j) {
            double likelihood = likelihood_matrix(i, j);
            cost_matrix(i, j) = -std::log(std::max(track_r * detection_probability_ * likelihood, 1e-12));
        }
        
        // Missed detection costs
        for (size_t j = 0; j < num_tracks; ++j) {
            if (i == j) {
                double miss_prob = (1.0 - track_r) + (track_r * (1.0 - detection_probability_));
                cost_matrix(i, num_meas + j) = -std::log(std::max(miss_prob, 1e-12));
            } else {
                cost_matrix(i, num_meas + j) = std::numeric_limits<double>::infinity();
            }
        }
    }
    
    // Step 3: Solve assignment (K-best)
    std::vector<Hypothesis> hypotheses = solve_assignment(cost_matrix, 100);
    
    // Normalize hypothesis weights (log-sum-exp)
    std::vector<double> log_weights;
    for (const auto& h : hypotheses) log_weights.push_back(-h.weight);
    double max_logw = *std::max_element(log_weights.begin(), log_weights.end());
    std::vector<double> norm_weights;
    double sum_exp = 0.0;
    for (double lw : log_weights) sum_exp += std::exp(lw - max_logw);
    for (double lw : log_weights) norm_weights.push_back(std::exp(lw - max_logw) / sum_exp);
    
    // Step 4: Combine, update and resample
    std::vector<Track> updated_tracks;
    for (size_t i = 0; i < num_tracks; ++i) {
        // Create a large mixture of particles from all hypothetical sets
        std::vector<Particle> mixed_particles;
        double new_existence_probability = 0.0;
        
        // For each hypothesis
        for (size_t h = 0; h < hypotheses.size(); ++h) {
            int assoc_idx = hypotheses[h].associations[i];
            double hyp_weight = norm_weights[h];
            
            // Select appropriate particle set based on association
            int particle_set_idx;
            if (assoc_idx >= num_meas) {
                // Missed detection case
                particle_set_idx = num_meas;
                
                // Update existence probability for missed detection
                double track_r = tracks[i].existence_probability();
                double miss_term = (1.0 - track_r) + track_r * (1.0 - detection_probability_);
                new_existence_probability += hyp_weight * miss_term;
            } else {
                // Measurement association case
                particle_set_idx = assoc_idx;
                
                // Update existence probability for detection
                double track_r = tracks[i].existence_probability();
                double detect_term = track_r * detection_probability_ * likelihood_matrix(i, assoc_idx);
                new_existence_probability += hyp_weight * detect_term;
            }
            
            // Add these particles to the mixture with appropriate weights
            for (const auto& particle : hypothetical_particle_sets[i][particle_set_idx]) {
                Particle mixed_particle;
                mixed_particle.state_vector = particle.state_vector;
                mixed_particle.weight = particle.weight * hyp_weight;
                mixed_particles.push_back(mixed_particle);
            }
        }
        
        // Normalize the weights in mixed particles
        double total_weight = 0.0;
        for (const auto& particle : mixed_particles) {
            total_weight += particle.weight;
        }
        
        if (total_weight > 1e-12) {
            for (auto& particle : mixed_particles) {
                particle.weight /= total_weight;
            }
        } else {
            for (auto& particle : mixed_particles) {
                particle.weight = 1.0 / mixed_particles.size();
            }
        }
        
        // Normalize the existence probability
        new_existence_probability = std::min(std::max(new_existence_probability, 0.0), 1.0);
        
        // Perform systematic resampling to get back to the standard number of particles
        size_t num_particles = tracks[i].particles().size();
        std::vector<Particle> resampled_particles;
        resampled_particles.reserve(num_particles);
        
        double u = ((double)rand() / RAND_MAX) / num_particles;
        double cumsum = 0.0;
        size_t idx = 0;
        
        for (size_t p = 0; p < num_particles; ++p) {
            double threshold = u + (double)p / num_particles;
            
            while (cumsum < threshold && idx < mixed_particles.size()) {
                cumsum += mixed_particles[idx].weight;
                ++idx;
            }
            
            // Get the chosen particle
            const Particle& chosen_particle = (idx > 0) ? mixed_particles[idx-1] : mixed_particles[0];
            
            // Create the resampled particle with equal weight
            Particle resampled;
            resampled.state_vector = chosen_particle.state_vector;
            resampled.weight = 1.0 / num_particles;
            resampled_particles.push_back(resampled);
        }
        
        // Create final track
        Track final_track = tracks[i];
        final_track.set_existence_probability(new_existence_probability);
        final_track.set_particles(resampled_particles);
        updated_tracks.push_back(final_track);
    }
    
    // Step 5: Handle birth
    // Find unused measurements from best hypothesis
    auto best_hyp_idx = std::max_element(norm_weights.begin(), norm_weights.end()) - norm_weights.begin();
    const auto& best_associations = hypotheses[best_hyp_idx].associations;
    
    std::vector<bool> used_meas(num_meas, false);
    for (int track_idx = 0; track_idx < best_associations.size(); ++track_idx) {
        int meas_idx = best_associations[track_idx];
        if (meas_idx >= 0 && meas_idx < num_meas) {
            used_meas[meas_idx] = true;
        }
    }
    
    std::vector<Measurement> unused_meas;
    for (size_t m = 0; m < num_meas; ++m) {
        if (!used_meas[m]) unused_meas.push_back(measurements[m]);
    }
    
    // Create new tracks from unused measurements
    std::vector<Track> born_tracks = birth_model_->generate_new_tracks(unused_meas, current_state_.timestamp());
    
    // Combine updated and born tracks
    updated_tracks.insert(updated_tracks.end(), born_tracks.begin(), born_tracks.end());
    
    // Track pruning: remove tracks with low existence probability
    const double prune_threshold = 0.01; // configurable
    std::vector<Track> pruned_tracks;
    for (const auto& track : updated_tracks) {
        if (track.existence_probability() >= prune_threshold) {
            pruned_tracks.push_back(track);
        }
    }
    
    // Update the current state with the new tracks
    current_state_.tracks().swap(pruned_tracks);
}

// Helper: average likelihood over all particles

double SMC_LMB_Tracker::compute_association_likelihood(const Track& track, const Measurement& measurement) const {
    double total_likelihood = 0.0;
    const auto& particles = track.particles();
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        double particle_likelihood = sensor_model_->calculate_likelihood(p, measurement);
        double weighted_likelihood = particle_likelihood * p.weight;
        total_likelihood += weighted_likelihood;
    }
    return total_likelihood;
}

const std::vector<Track>& SMC_LMB_Tracker::get_tracks() const {
    return current_state_.tracks();
}

void SMC_LMB_Tracker::set_tracks(const std::vector<Track>& tracks) {
    current_state_.set_tracks(tracks);
}