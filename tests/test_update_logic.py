import numpy as np
import sys
import os

# Ensure lmb_engine.pyd is found in the build output directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/lmb_engine/Debug')))
import lmb_engine

def test_compute_association_likelihood():
    propagator = lmb_engine.LinearPropagator()
    sensor_model = lmb_engine.SimpleSensorModel()
    birth_model = lmb_engine.AdaptiveBirthModel(10, 0.1, np.eye(7))
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)
    # Create 3 particles with varied weights
    particles = []
    for state, weight in zip([[10,0,0,0,0,0], [100,0,0,0,0,0], [200,0,0,0,0,0]], [0.5, 0.3, 0.2]):
        p = lmb_engine.Particle()
        p.state_vector = np.array(state)
        p.weight = weight
        particles.append(p)
    track = lmb_engine.Track()
    track.set_particles(particles)
    measurement = lmb_engine.Measurement()
    measurement.value_ = np.array([10,0,0,0,0,0])
    measurement.covariance_ = np.eye(6)
    actual_likelihood = tracker.compute_association_likelihood(track, measurement)
    assert actual_likelihood > 0.0, "Test 1 Failed: Likelihood should be positive."
    assert actual_likelihood < 1.0, "Test 1 Failed: Likelihood seems unreasonably high."
    print("Test 1 Passed")

def test_solve_assignment():
    cost_matrix = np.array([
        [1.0, 100.0, 10.0],
        [100.0, 2.0, 10.0]
    ])
    hypotheses = lmb_engine.solve_assignment(cost_matrix, 1)
    assert len(hypotheses) == 1, "Test 2 failed: wrong number of hypotheses"
    assert hypotheses[0].associations == [0, 1], f"Test 2 failed: associations {hypotheses[0].associations} != [0, 1]"
    assert abs(hypotheses[0].weight - 3.0) < 1e-9, f"Test 2 failed: weight {hypotheses[0].weight} != 3.0"
    print("Test 2 Passed")

def test_full_update_logic():
    propagator = lmb_engine.LinearPropagator()
    sensor_model = lmb_engine.SimpleSensorModel()
    birth_model = lmb_engine.AdaptiveBirthModel(10, 0.1, np.eye(7))
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)
    # Track 0 with varied weights
    particles_t0 = []
    for state, weight in zip([[10,0,0,0,0,0], [100,0,0,0,0,0], [200,0,0,0,0,0]], [0.5, 0.3, 0.2]):
        p = lmb_engine.Particle()
        p.state_vector = np.array(state)
        p.weight = weight
        particles_t0.append(p)
    track0 = lmb_engine.Track()
    track0.set_particles(particles_t0)
    # Track 1 with varied weights
    particles_t1 = []
    for state, weight in zip([[20,0,0,0,0,0], [120,0,0,0,0,0], [220,0,0,0,0,0]], [0.5, 0.3, 0.2]):
        p = lmb_engine.Particle()
        p.state_vector = np.array(state)
        p.weight = weight
        particles_t1.append(p)
    track1 = lmb_engine.Track()
    track1.set_particles(particles_t1)
    # Measurements
    meas0 = lmb_engine.Measurement()
    meas0.value_ = np.array([10,0,0,0,0,0])
    meas0.covariance_ = np.eye(6)
    meas1 = lmb_engine.Measurement()
    meas1.value_ = np.array([20,0,0,0,0,0])
    meas1.covariance_ = np.eye(6)
    measurements = [meas0, meas1]
    tracker.set_tracks([track0, track1])
    tracker.update(measurements)
    updated_tracks = tracker.get_tracks()
    # Check weights: high-weight particle should have greater weight than far particle
    t0_particles = updated_tracks[0].particles()
    t1_particles = updated_tracks[1].particles()
    weights_t0 = [p.weight for p in t0_particles]
    weights_t1 = [p.weight for p in t1_particles]
    assert weights_t0[0] > weights_t0[1], "Test 3 failed: Track 0 weights not updated correctly"
    assert weights_t1[0] > weights_t1[1], "Test 3 failed: Track 1 weights not updated correctly"
    print("Test 3 Passed")

if __name__ == "__main__":
    test_compute_association_likelihood()
    test_solve_assignment()
    test_full_update_logic()
    print("ALL UNIT TESTS PASSED")
