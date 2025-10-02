#include "smc_lmb_tracker.h"
#include "assignment.h"

SMC_LMB_Tracker::SMC_LMB_Tracker(std::shared_ptr<IOrbitPropagator> propagator,
                                 std::shared_ptr<ISensorModel> sensor_model,
                                 std::shared_ptr<IBirthModel> birth_model,
                                 double survival_probability)
    : current_state_(0.0, std::vector<Track>{}),  // Initialize with timestamp 0.0 and empty track list
      propagator_(std::move(propagator)),
      sensor_model_(std::move(sensor_model)),
      birth_model_(std::move(birth_model)),
      survival_probability_(survival_probability) {
}

void SMC_LMB_Tracker::predict(double dt) {
    // Projects the filter state forward in time.
    current_state_.set_timestamp(current_state_.timestamp() + dt);
    
    // Get a mutable copy of tracks for modification
    std::vector<Track> tracks = current_state_.tracks();
    
    for (Track& track : tracks) {
        // Update existence probability by applying survival probability
        track.set_existence_probability(track.existence_probability() * survival_probability_);
        
        // Propagate particles
        std::vector<Particle> propagated_particles;
        propagated_particles.reserve(track.particles().size());
        
        for (const Particle& particle : track.particles()) {
            Particle new_particle = propagator_->propagate(particle, dt);
            propagated_particles.push_back(new_particle);
        }
        
        // Replace track's old particle cloud with propagated particles
        track.set_particles(propagated_particles);
    }
    
    // Set the modified tracks back to the state
    current_state_.set_tracks(tracks);
}

void SMC_LMB_Tracker::update(const std::vector<Measurement>& measurements) {
    std::vector<Track> tracks = current_state_.tracks();
    size_t num_tracks = tracks.size();
    size_t num_meas = measurements.size();
    if (num_tracks == 0) {
        std::vector<Track> born_tracks = birth_model_->generate_new_tracks(measurements, current_state_.timestamp());
        set_tracks(born_tracks);
        return;
    }
    // 1. Build cost matrix (tracks x (measurements + 1))
    Eigen::MatrixXd cost_matrix(num_tracks, num_meas + 1);
    for (size_t i = 0; i < num_tracks; ++i) {
        for (size_t j = 0; j < num_meas; ++j) {
            double likelihood = compute_association_likelihood(tracks[i], measurements[j]);
            cost_matrix(i, j) = -std::log(std::max(likelihood, 1e-12));
        }
        cost_matrix(i, num_meas) = -std::log(std::max(1.0 - tracks[i].existence_probability(), 1e-12));
    }
    // 2. Solve assignment (K-best)
    std::vector<Hypothesis> hypotheses = solve_assignment(cost_matrix, 100);
    // 3. Normalize weights (log-sum-exp)
    std::vector<double> log_weights;
    for (const auto& h : hypotheses) log_weights.push_back(-h.weight);
    double max_logw = *std::max_element(log_weights.begin(), log_weights.end());
    std::vector<double> norm_weights;
    double sum_exp = 0.0;
    for (double lw : log_weights) sum_exp += std::exp(lw - max_logw);
    for (double lw : log_weights) norm_weights.push_back(std::exp(lw - max_logw) / sum_exp);
    // 4. Final LMB Update
    std::vector<Track> updated_tracks;
    for (size_t track_idx = 0; track_idx < num_tracks; ++track_idx) {
        const auto& current_particles = tracks[track_idx].particles();
        size_t num_particles = current_particles.size();
        double miss_detection_sum = 0.0;
        std::vector<double> measurement_association_sums(num_meas, 0.0);
        std::vector<std::vector<double>> measurement_particle_likelihoods(num_meas, std::vector<double>(num_particles));
        for (size_t m = 0; m < num_meas; ++m) {
            for (size_t p = 0; p < num_particles; ++p) {
                measurement_particle_likelihoods[m][p] = sensor_model_->calculate_likelihood(current_particles[p], measurements[m]);
            }
        }
        for (size_t h = 0; h < hypotheses.size(); ++h) {
            int assoc_meas_idx = hypotheses[h].associations[track_idx];
            if (assoc_meas_idx >= num_meas) {
                miss_detection_sum += norm_weights[h];
            } else {
                measurement_association_sums[assoc_meas_idx] += norm_weights[h];
            }
        }
        double miss_term = miss_detection_sum * tracks[track_idx].existence_probability();
        double new_existence_probability = miss_term;
        std::vector<double> final_particle_weights(num_particles, 0.0);
        for (size_t p = 0; p < num_particles; ++p) {
            final_particle_weights[p] = miss_term * current_particles[p].weight;
        }
        for (size_t m = 0; m < num_meas; ++m) {
            if (measurement_association_sums[m] > 0.0) {
                double assoc_term = measurement_association_sums[m] * (1.0 - tracks[track_idx].existence_probability());
                new_existence_probability += assoc_term;
                for (size_t p = 0; p < num_particles; ++p) {
                    double particle_likelihood = measurement_particle_likelihoods[m][p];
                    double increment = assoc_term * current_particles[p].weight * particle_likelihood;
                    final_particle_weights[p] += increment;
                }
            }
        }
        assert(final_particle_weights.size() == current_particles.size());
        std::vector<Particle> updated_particles = current_particles;
        double total_weight = 0.0;
        for (size_t p = 0; p < num_particles; ++p) {
            total_weight += final_particle_weights[p];
        }
        if (total_weight > 1e-12) {
            for (size_t p = 0; p < num_particles; ++p) {
                updated_particles[p].weight = final_particle_weights[p] / total_weight;
            }
        } else {
            for (size_t p = 0; p < num_particles; ++p) {
                updated_particles[p].weight = 1.0 / num_particles;
            }
        }
        double sum_weights = 0.0;
        // Systematic resampling
        std::vector<Particle> resampled_particles;
        double u = ((double)rand() / RAND_MAX) / num_particles;
        double cumsum = 0.0;
        size_t idx = 0;
        for (size_t i = 0; i < num_particles; ++i) {
            double threshold = u + (double)i / num_particles;
            while (cumsum < threshold && idx < num_particles) {
                cumsum += updated_particles[idx].weight;
                ++idx;
            }
            Particle resampled = (idx > 0) ? updated_particles[idx-1] : updated_particles[0];
            resampled.weight = 1.0 / num_particles;
            resampled_particles.push_back(resampled);
        }
        Track final_track = tracks[track_idx];
        final_track.set_existence_probability(new_existence_probability);
        final_track.set_particles(resampled_particles);
        updated_tracks.push_back(final_track);
    }
    // 5. Find unused measurements from best hypothesis
    auto best_hyp = std::max_element(norm_weights.begin(), norm_weights.end()) - norm_weights.begin();
    std::vector<bool> used_meas(num_meas, false);
    for (int assoc : hypotheses[best_hyp].associations) {
        if (assoc != num_meas) used_meas[assoc] = true;
    }
    std::vector<Measurement> unused_meas;
    for (size_t m = 0; m < num_meas; ++m) {
        if (!used_meas[m]) unused_meas.push_back(measurements[m]);
    }
    std::vector<Track> born_tracks = birth_model_->generate_new_tracks(unused_meas, current_state_.timestamp());
    // 6. Combine updated and born tracks
    updated_tracks.insert(updated_tracks.end(), born_tracks.begin(), born_tracks.end());
    // Track pruning: remove tracks with low existence probability
    const double prune_threshold = 0.01; // configurable
    std::vector<Track> pruned_tracks;
    for (const auto& track : updated_tracks) {
        if (track.existence_probability() >= prune_threshold) {
            pruned_tracks.push_back(track);
        }
    }

    set_tracks(pruned_tracks);
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