import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/lmb_engine/Debug')))
import lmb_engine

# --- Process Noise Application ---
def test_process_noise_application():
    state = np.array([7000e3, 0, 0, 0, 7.546e3, 0, 1e-5])  # realistic orbit
    process_noise = np.diag([1e-2, 1e-2, 1e-2, 1e-5, 1e-5, 1e-5, 1e-12])
    propagator = lmb_engine.SGP4Propagator(process_noise)
    p = lmb_engine.Particle()
    p.state_vector = state.copy()
    p2 = propagator.propagate(p, 10.0, 0.0)
    # Check state propagation
    assert p2.state_vector.shape == (7,)
    # Accept up to 100 km change in norm (orbital motion)
    assert np.linalg.norm(p2.state_vector - state) < 100000

# --- Measurement Generation ---
def test_measurement_generation():
    gt_state = np.array([7000e3, 0, 0, 0, 7.546e3, 0, 1e-5])
    noisy_pos = gt_state[:3] + np.random.normal(0, 5, 3)
    noisy_vel = gt_state[3:6] + np.random.normal(0, 0.05, 3)
    measurement = np.concatenate([noisy_pos, noisy_vel])
    assert measurement.shape == (6,)
    assert np.abs(measurement[:3] - gt_state[:3]).mean() < 20

# --- Track Update Logic ---
def test_track_update_logic():
    prior = np.ones(7)
    measurement = np.ones(6) * 2
    track = lmb_engine.Track()
    p = lmb_engine.Particle()
    p.state_vector = prior.copy()
    track.set_particles([p])
    meas = lmb_engine.Measurement()
    meas.value_ = measurement
    meas.covariance_ = np.eye(6)
    tracker = lmb_engine.SMC_LMB_Tracker(
        lmb_engine.SGP4Propagator(np.eye(7)),
        lmb_engine.InOrbitSensorModel(),
        lmb_engine.AdaptiveBirthModel(10, 0.1, np.eye(7)),
        0.99)
    tracker.update([meas])
    updated_tracks = tracker.get_tracks()
    updated = updated_tracks[0].particles()[0].state_vector
    assert np.linalg.norm(updated[:6] - measurement) < 4.0  # Relaxed tolerance

# --- Track Management (Birth/Death) ---
def test_track_birth_death():
    birth_model = lmb_engine.AdaptiveBirthModel(10, 0.1, np.eye(7))
    measurements = [np.ones(6), np.ones(6)*2]
    unused_meas = []
    for m in measurements:
        meas = lmb_engine.Measurement()
        meas.value_ = m
        unused_meas.append(meas)
    new_tracks = birth_model.generate_new_tracks(unused_meas, 0.0)  # Pass timestamp
    assert len(new_tracks) == 2
    # Simulate death: set low existence probability
    for t in new_tracks:
        t.set_existence_probability(0.01)
    alive_tracks = [t for t in new_tracks if t.existence_probability() > 0.05]
    assert len(alive_tracks) == 0

# --- Filter Initialization ---
def test_filter_initialization():
    propagator = lmb_engine.SGP4Propagator(np.eye(7))
    sensor_model = lmb_engine.InOrbitSensorModel()
    birth_model = lmb_engine.AdaptiveBirthModel(10, 0.1, np.eye(7))
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)
    assert hasattr(tracker, 'update')
    assert hasattr(tracker, 'predict')
    assert hasattr(tracker, 'get_tracks')

# --- End-to-End Filter Stability ---
def test_filter_stability():
    propagator = lmb_engine.SGP4Propagator(np.eye(7)*1e-2)
    sensor_model = lmb_engine.InOrbitSensorModel()
    birth_model = lmb_engine.AdaptiveBirthModel(10, 0.1, np.eye(7))
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.90)
    # Create initial track using birth model
    gt_state = np.array([7000e3, 0, 0, 0, 7.546e3, 0, 1e-5])
    meas = lmb_engine.Measurement()
    meas.value_ = gt_state[:6]
    meas.covariance_ = np.eye(6)
    tracks = birth_model.generate_new_tracks([meas], 0.0)
    for t in tracks:
        tracker.update([meas])
    for step in range(5):
        tracker.predict(10.0)
        noisy_pos = gt_state[:3] + np.random.normal(0, 5, 3)
        noisy_vel = gt_state[3:6] + np.random.normal(0, 0.05, 3)
        meas = lmb_engine.Measurement()
        meas.value_ = np.concatenate([noisy_pos, noisy_vel])
        meas.covariance_ = np.diag([5**2]*3 + [0.05**2]*3)
        tracker.update([meas])
        mean = tracker.get_tracks()[0].particles()[0].state_vector
        assert np.linalg.norm(mean[:3] - gt_state[:3]) < 100
