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
    // TODO: Implement prediction step
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