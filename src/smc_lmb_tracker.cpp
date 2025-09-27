#include "smc_lmb_tracker.h"

SMC_LMB_Tracker::SMC_LMB_Tracker(std::unique_ptr<IOrbitPropagator> propagator,
                                 std::unique_ptr<ISensorModel> sensor_model,
                                 std::unique_ptr<IBirthModel> birth_model,
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
    // TODO: Implement update step
}

const std::vector<Track>& SMC_LMB_Tracker::get_tracks() const {
    return current_state_.tracks();
}

void SMC_LMB_Tracker::set_tracks(const std::vector<Track>& tracks) {
    current_state_.set_tracks(tracks);
}