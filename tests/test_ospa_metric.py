import sys
import os
import numpy as np
# Ensure lmb_engine is importable (Debug build path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/lmb_engine/Debug')))
import lmb_engine

def make_particle(pos):
    p = lmb_engine.Particle()
    # Set the state vector and cartesian_state_vector for testing
    full_state = np.array([pos[0], pos[1], pos[2], 0, 0, 0, 0], dtype=float)
    p.state_vector = full_state
    p.cartesian_state_vector = np.array(pos, dtype=float)
    p.weight = 1.0
    return p

def make_track(pos):
    particles = [make_particle(pos)]
    label = lmb_engine.TrackLabel()
    label.birth_time = 0
    label.index = 0
    return lmb_engine.Track(label, 1.0, particles)

def test_ospa_perfect_match():
    tracks = [make_track(np.array([1.0, 2.0, 3.0]))]
    truths = [np.array([1.0, 2.0, 3.0])]
    ospa = lmb_engine.calculate_ospa_distance(tracks, truths, 100.0)
    print(f"OSPA (perfect match): {ospa}")
    assert abs(ospa) < 1e-6, "OSPA should be zero for perfect match"

def test_ospa_offset():
    tracks = [make_track(np.array([1.0, 2.0, 3.0]))]
    truths = [np.array([4.0, 6.0, 3.0])]
    ospa = lmb_engine.calculate_ospa_distance(tracks, truths, 100.0)
    expected = np.linalg.norm(np.array([1.0,2.0,3.0]) - np.array([4.0,6.0,3.0]))
    print(f"OSPA (offset): {ospa}, expected: {expected}")
    assert abs(ospa - expected) < 1e-6, "OSPA should equal Euclidean distance for single track/truth"

def test_ospa_cutoff():
    tracks = [make_track(np.array([0.0, 0.0, 0.0]))]
    truths = [np.array([1000.0, 0.0, 0.0])]
    ospa = lmb_engine.calculate_ospa_distance(tracks, truths, 10.0)
    print(f"OSPA (cutoff): {ospa}")
    assert abs(ospa - 10.0) < 1e-6, "OSPA should be cutoff value when distance exceeds cutoff"

if __name__ == "__main__":
    test_ospa_perfect_match()
    test_ospa_offset()
    test_ospa_cutoff()
    print("All OSPA tests passed.")
