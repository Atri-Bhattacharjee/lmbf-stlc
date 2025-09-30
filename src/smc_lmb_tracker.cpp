#include "smc_lmb_tracker.h"

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
    // Get current tracks
    std::vector<Track> tracks = current_state_.tracks();
    size_t num_tracks = tracks.size();
    size_t num_meas = measurements.size();
    if (num_tracks == 0) {
        // No tracks: call birth model on all measurements
        std::vector<Track> born_tracks = birth_model_->generate_new_tracks(measurements, current_state_.timestamp());
        set_tracks(born_tracks);
        return;
    }

    // Generate all possible hypotheses
    std::vector<Hypothesis> hypotheses;
    std::vector<int> current_association(num_tracks, -1);
    std::vector<bool> meas_is_used(num_meas, false);
    generate_hypotheses_recursive(0, current_association, meas_is_used, tracks, measurements, hypotheses);

    // Normalize hypothesis weights
    double total_weight = 0.0;
    for (const auto& h : hypotheses) total_weight += h.weight;
    if (total_weight > 0.0) {
        for (auto& h : hypotheses) h.weight /= total_weight;
    }

    // LMB moment-matching update
    std::vector<Track> updated_tracks;
    for (size_t t = 0; t < num_tracks; ++t) {
        // New existence probability: sum of weights for hypotheses where track t is detected
        double new_ex_prob = 0.0;
        for (const auto& h : hypotheses) {
            if (h.associations[t] != -1) new_ex_prob += h.weight;
        }
        // New particles: weighted mixture over all hypotheses
        std::vector<Particle> new_particles;
        const auto& old_particles = tracks[t].particles();
        for (size_t p = 0; p < old_particles.size(); ++p) {
            Particle updated_p = old_particles[p];
            double w_sum = 0.0;
            for (const auto& h : hypotheses) {
                double w = h.weight;
                if (h.associations[t] != -1) {
                    // Associated: multiply by likelihood
                    w *= compute_association_likelihood(tracks[t], measurements[h.associations[t]]);
                } else {
                    // Missed detection: use survival probability only
                    w *= tracks[t].existence_probability();
                }
                w_sum += w;
            }
            updated_p.weight = w_sum;
            new_particles.push_back(updated_p);
        }
        // Normalize particle weights
        double norm = 0.0;
        for (const auto& p : new_particles) norm += p.weight;
        if (norm > 0.0) {
            for (auto& p : new_particles) p.weight /= norm;
        }
        // Systematic resampling
        std::vector<Particle> resampled;
        size_t N = new_particles.size();
        double u = ((double)rand() / RAND_MAX) / N;
        double cumsum = 0.0;
        size_t idx = 0;
        for (size_t i = 0; i < N; ++i) {
            double threshold = u + (double)i / N;
            while (cumsum < threshold && idx < N) {
                cumsum += new_particles[idx].weight;
                ++idx;
            }
            if (idx > 0) resampled.push_back(new_particles[idx-1]);
            else resampled.push_back(new_particles[0]);
        }
        Track updated_track = tracks[t];
        updated_track.set_existence_probability(new_ex_prob);
        updated_track.set_particles(resampled);
        updated_tracks.push_back(updated_track);
    }

    // Find unused measurements from the highest-weight hypothesis
    auto best_hyp = std::max_element(hypotheses.begin(), hypotheses.end(), [](const Hypothesis& a, const Hypothesis& b) { return a.weight < b.weight; });
    std::vector<bool> used_meas(num_meas, false);
    if (best_hyp != hypotheses.end()) {
        for (int assoc : best_hyp->associations) {
            if (assoc != -1) used_meas[assoc] = true;
        }
    }
    std::vector<Measurement> unused_meas;
    for (size_t m = 0; m < num_meas; ++m) {
        if (!used_meas[m]) unused_meas.push_back(measurements[m]);
    }
    std::vector<Track> born_tracks = birth_model_->generate_new_tracks(unused_meas, current_state_.timestamp());

    // Combine updated and born tracks
    updated_tracks.insert(updated_tracks.end(), born_tracks.begin(), born_tracks.end());
    set_tracks(updated_tracks);
}

// Helper: average likelihood over all particles

double SMC_LMB_Tracker::compute_association_likelihood(const Track& track, const Measurement& measurement) const {
    double likelihood = 0.0;
    const auto& particles = track.particles();
    for (const auto& p : particles) {
        likelihood += sensor_model_->calculate_likelihood(p, measurement) * p.weight;
    }
    return likelihood;
}

// Helper: recursively generate all hypotheses
void SMC_LMB_Tracker::generate_hypotheses_recursive(
    int track_idx,
    std::vector<int>& current_association,
    std::vector<bool>& meas_is_used,
    const std::vector<Track>& tracks,
    const std::vector<Measurement>& measurements,
    std::vector<Hypothesis>& out_hypotheses) const {
    size_t num_tracks = tracks.size();
    size_t num_meas = measurements.size();
    if (track_idx == num_tracks) {
        // Base case: all tracks assigned
        double weight = 1.0;
        for (size_t t = 0; t < num_tracks; ++t) {
            if (current_association[t] == -1) {
                // Missed detection
                weight *= tracks[t].existence_probability();
            } else {
                // Associated
                weight *= compute_association_likelihood(tracks[t], measurements[current_association[t]]);
            }
        }
        Hypothesis h;
        h.associations = current_association;
        h.weight = weight;
        out_hypotheses.push_back(h);
        return;
    }
    // Missed detection case
    current_association[track_idx] = -1;
    generate_hypotheses_recursive(track_idx + 1, current_association, meas_is_used, tracks, measurements, out_hypotheses);
    // Try all unused measurements
    for (size_t m = 0; m < num_meas; ++m) {
        if (!meas_is_used[m]) {
            current_association[track_idx] = (int)m;
            meas_is_used[m] = true;
            generate_hypotheses_recursive(track_idx + 1, current_association, meas_is_used, tracks, measurements, out_hypotheses);
            meas_is_used[m] = false; // backtrack
        }
    }
}

const std::vector<Track>& SMC_LMB_Tracker::get_tracks() const {
    return current_state_.tracks();
}

void SMC_LMB_Tracker::set_tracks(const std::vector<Track>& tracks) {
    current_state_.set_tracks(tracks);
}