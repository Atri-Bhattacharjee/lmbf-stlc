import sys
import os
import numpy as np

# Ensure lmb_engine is importable (Debug build path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/lmb_engine/Debug')))
import lmb_engine

def test_assignment_solver():
    # Justification: This tests the assignment solver with a simple cost matrix.
    # Setup: 2 tracks, 2 measurements, plus missed detection column.
    cost_matrix = np.array([
        [1.0, 2.0, 5.0],  # Track 0: best to meas 0
        [2.0, 1.0, 5.0],  # Track 1: best to meas 1
    ])
    # Call the assignment solver for 2-best assignments
    results = lmb_engine.solve_assignment(cost_matrix, 2)
    print("Assignment solver results:")
    for hyp in results:
        print("Associations:", hyp.associations, "Cost:", hyp.weight)
    # Assert the best assignment is [0, 1] (track 0->meas 0, track 1->meas 1)
    assert results[0].associations == [0, 1], "Best assignment should be [0, 1]"

def test_tracker_update():
    # Justification: This tests the tracker update with a minimal scenario.
    # Setup: 1 track, 1 measurement, perfect likelihood.
    tracker = lmb_engine.SMC_LMB_Tracker()
    # Create a dummy propagator, sensor model, and birth model
    propagator = lmb_engine.LinearPropagator()
    sensor_model = lmb_engine.SimpleSensorModel()
    covariance = np.eye(7) * 0.1
    birth_model = lmb_engine.AdaptiveBirthModel(10, 0.9, covariance)
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)
    # Create a track with one particle
    particle = lmb_engine.Particle()
    particle.cartesian_state_vector = np.array([7000, 0, 0, 0, 7.5, 0, 0])
    particle.weight = 1.0
    track_label = lmb_engine.TrackLabel()
    track = lmb_engine.Track(track_label, 0.9, [particle])
    tracker.set_tracks([track])
    # Create a measurement
    measurement = lmb_engine.Measurement()
    measurement.value_ = np.array([0.0, 0.0, 10.0])
    measurement.covariance_ = np.eye(3) * 1e-3
    measurement.sensor_id_ = "sensor_1"
    measurement.sensor_state_ = np.array([6990, 0, 0, 0, 7.5, 0])
    # Run update
    tracker.update([measurement])
    tracks = tracker.get_tracks()
    print("Tracker update results:")
    for t in tracks:
        print("Track existence probability:", t.existence_probability())
        for p in t.particles():
            print("Particle weight:", p.weight)
    # Assert at least one track exists and weights are normalized
    assert len(tracks) > 0, "Tracker should have at least one track after update"
    assert abs(sum(p.weight for p in tracks[0].particles()) - 1.0) < 1e-6, "Particle weights should be normalized"

if __name__ == "__main__":
    test_assignment_solver()
    test_tracker_update()
    print("All tests passed!")