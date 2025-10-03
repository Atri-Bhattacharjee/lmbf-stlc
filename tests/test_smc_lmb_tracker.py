import numpy as np
import sys
import os

# Ensure lmb_engine.pyd is found in the build output directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/lmb_engine/Debug')))
import lmb_engine

def make_track(state, existence=0.9, num_particles=5):
    particles = []
    for _ in range(num_particles):
        p = lmb_engine.Particle()
        p.state_vector = np.array(state, dtype=float)
        p.weight = 1.0 / num_particles
        particles.append(p)
    label = lmb_engine.TrackLabel()
    label.birth_time = 0
    label.index = 0
    return lmb_engine.Track(label, existence, particles)

def make_measurement(state):
    meas = lmb_engine.Measurement()
    meas.timestamp_ = 0.0
    meas.value_ = np.array(state, dtype=float)
    meas.covariance_ = np.eye(len(state)) * 0.01
    meas.sensor_id_ = "sensor"
    meas.sensor_state_ = np.array(state[:6], dtype=float)
    return meas

def setup_tracker():
    # Use minimal noise for deterministic behavior
    process_noise = np.eye(7) * 1e-8
    propagator = lmb_engine.SGP4Propagator(process_noise)
    sensor_model = lmb_engine.InOrbitSensorModel()
    birth_cov = np.eye(7) * 0.01
    birth_model = lmb_engine.AdaptiveBirthModel(5, 0.1, birth_cov)
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)
    return tracker

def test_perfect_match():
    print("Test: Perfect match (2 tracks, 2 measurements)")
    tracker = setup_tracker()
    state1 = [7000e3, 0, 0, 0, 7.5e3, 10, 1e-4]
    state2 = [7001e3, 0, 0, 0, 7.5e3, -10, 1e-4]
    track1 = make_track(state1)
    track2 = make_track(state2)
    tracker.set_tracks([track1, track2])
    meas1 = make_measurement(state1)
    meas2 = make_measurement(state2)
    tracker.update([meas1, meas2])
    tracks = tracker.get_tracks()
    print(f"  Number of tracks after update: {len(tracks)}")
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"
    print("  Track states:")
    for t in tracks:
        print(f"    {t.particles()[0].state_vector}")
    print("  PASS\n")

def test_track_measurement_mismatch():
    print("Test: Mismatch (2 tracks, 2 measurements, no match)")
    tracker = setup_tracker()
    state1 = [7000e3, 0, 0, 0, 7.5e3, 10, 1e-4]
    state2 = [7001e3, 0, 0, 0, 7.5e3, -10, 1e-4]
    track1 = make_track(state1)
    track2 = make_track(state2)
    tracker.set_tracks([track1, track2])
    meas1 = make_measurement([8000e3, 0, 0, 0, 7.5e3, 10, 1e-4])
    meas2 = make_measurement([8001e3, 0, 0, 0, 7.5e3, -10, 1e-4])
    tracker.update([meas1, meas2])
    tracks = tracker.get_tracks()
    print(f"  Number of tracks after update: {len(tracks)}")
    assert len(tracks) == 4, f"Expected 4 tracks (2 old, 2 new), got {len(tracks)}"
    print("  PASS\n")

def test_imbalance_cases():
    print("Test: Imbalance (2 tracks, 1 measurement)")
    tracker = setup_tracker()
    state1 = [7000e3, 0, 0, 0, 7.5e3, 10, 1e-4]
    state2 = [7001e3, 0, 0, 0, 7.5e3, -10, 1e-4]
    track1 = make_track(state1)
    track2 = make_track(state2)
    tracker.set_tracks([track1, track2])
    meas1 = make_measurement(state1)
    tracker.update([meas1])
    tracks = tracker.get_tracks()
    print(f"  Number of tracks after update: {len(tracks)}")
    assert len(tracks) == 2 or len(tracks) == 3, f"Expected 2 or 3 tracks, got {len(tracks)}"
    print("  PASS\n")
    print("Test: Imbalance (1 track, 2 measurements)")
    tracker = setup_tracker()
    track1 = make_track(state1)
    tracker.set_tracks([track1])
    meas1 = make_measurement(state1)
    meas2 = make_measurement(state2)
    tracker.update([meas1, meas2])
    tracks = tracker.get_tracks()
    print(f"  Number of tracks after update: {len(tracks)}")
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"
    print("  PASS\n")

def test_no_measurements():
    print("Test: No measurements (2 tracks, 0 measurements)")
    tracker = setup_tracker()
    state1 = [7000e3, 0, 0, 0, 7.5e3, 10, 1e-4]
    state2 = [7001e3, 0, 0, 0, 7.5e3, -10, 1e-4]
    track1 = make_track(state1)
    track2 = make_track(state2)
    tracker.set_tracks([track1, track2])
    tracker.update([])
    tracks = tracker.get_tracks()
    print(f"  Number of tracks after update: {len(tracks)}")
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"
    print("  PASS\n")

def test_no_tracks():
    print("Test: No tracks (0 tracks, 2 measurements)")
    tracker = setup_tracker()
    meas1 = make_measurement([7000e3, 0, 0, 0, 7.5e3, 10, 1e-4])
    meas2 = make_measurement([7001e3, 0, 0, 0, 7.5e3, -10, 1e-4])
    tracker.update([meas1, meas2])
    tracks = tracker.get_tracks()
    print(f"  Number of tracks after update: {len(tracks)}")
    assert len(tracks) == 2, f"Expected 2 tracks, got {len(tracks)}"
    print("  PASS\n")

def main():
    test_perfect_match()
    test_track_measurement_mismatch()
    test_imbalance_cases()
    test_no_measurements()
    test_no_tracks()
    print("All SMC_LMB_Tracker tests passed.")

if __name__ == "__main__":
    main()
