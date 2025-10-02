import numpy as np
import sys
import os

# --- Robust module finding block ---
engine_debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'lmb_engine', 'Debug'))
if os.path.isdir(engine_debug_dir):
    sys.path.append(engine_debug_dir)
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
debug_build_dir = os.path.join(build_dir, 'Debug')
release_build_dir = os.path.join(build_dir, 'Release')
if os.path.isdir(debug_build_dir):
    sys.path.append(debug_build_dir)
elif os.path.isdir(release_build_dir):
    sys.path.append(release_build_dir)
else:
    sys.path.append(build_dir)
try:
    import lmb_engine
    print(f"Imported lmb_engine from: {lmb_engine.__file__}")
except ImportError:
    print("FATAL ERROR: Could not import the 'lmb_engine' module.")
    print(f"Ensure the compiled C++ module (.pyd file) is in one of these paths:")
    print(f"  - {engine_debug_dir}")
    print(f"  - {debug_build_dir}")
    print(f"  - {release_build_dir}")
    print(f"  - {build_dir}")
    sys.exit(1)


# Helper to print particles and weights
def print_particles(track, label):
    ps = track.particles()
    print(f"{label} (N={len(ps)}):")
    for i, p in enumerate(ps):
        print(f"  Particle {i}: state={p.state_vector}, weight={p.weight}")

# Create a tracker with 2 tracks, each with 5 particles
num_particles = 5
state_dim = 6

# Initial states for two tracks
states = [
    np.array([7000e3, 0, 0, 0, 7.546e3, 0]),
    np.array([7000e3, 100e3, 0, 0, 7.546e3, 0])
]

# Create particles for each track, spaced far apart
tracks = []
for i, s in enumerate(states):
    particles = []
    for j in range(num_particles):
        offset = np.zeros(state_dim)
        offset[0] = j * 1e5  # space particles 100 km apart in x
        p = lmb_engine.Particle()
        p.state_vector = s + offset
        p.weight = 1.0 / num_particles
        particles.append(p)
    track = lmb_engine.Track()
    track.set_particles(particles)
    track.set_existence_probability(0.9)
    tracks.append(track)

# Create tracker
tracker = lmb_engine.create_custom_smc_lmb_tracker(num_particles, 0.9, np.eye(7)*0.1, 0.99)
tracker.set_tracks(tracks)

# --- BEGIN CONSISTENT MEASUREMENT GENERATION ---
# Define noise parameters
pos_noise_std = 5.0  # meters
vel_noise_std = 0.05  # m/s

# Create a consistent covariance matrix
covariance_matrix = np.diag([pos_noise_std**2]*3 + [vel_noise_std**2]*3)

# Generate noise for Measurement 0
pos_noise_0 = np.random.normal(0, pos_noise_std, 3)
vel_noise_0 = np.random.normal(0, vel_noise_std, 3)
noise_vector_0 = np.concatenate([pos_noise_0, vel_noise_0])

meas0 = lmb_engine.Measurement()
meas0.value_ = tracks[0].particles()[0].state_vector + noise_vector_0
meas0.covariance_ = covariance_matrix
meas0.timestamp_ = 0.0
meas0.sensor_id_ = "0"

# Generate noise for Measurement 1
pos_noise_1 = np.random.normal(0, pos_noise_std, 3)
vel_noise_1 = np.random.normal(0, vel_noise_std, 3)
noise_vector_1 = np.concatenate([pos_noise_1, vel_noise_1])

meas1 = lmb_engine.Measurement()
meas1.value_ = tracks[1].particles()[0].state_vector + noise_vector_1
meas1.covariance_ = covariance_matrix
meas1.timestamp_ = 0.0
meas1.sensor_id_ = "1"

measurements = [meas0, meas1]
# --- END CONSISTENT MEASUREMENT GENERATION ---

# Print before update
print("=== BEFORE UPDATE ===")
for i, t in enumerate(tracker.get_tracks()):
    print_particles(t, f"Track {i}")
    print(f"  Existence probability: {t.existence_probability()}")

# Run update
tracker.update(measurements)

# Print after update, before resampling
print("=== AFTER UPDATE (before resampling) ===")
for i, t in enumerate(tracker.get_tracks()):
    print_particles(t, f"Track {i}")
    print(f"  Existence probability: {t.existence_probability()}")

# Run predict to trigger resampling (if not already done)
tracker.predict(1.0)

# Print after resampling
print("=== AFTER RESAMPLING ===")
for i, t in enumerate(tracker.get_tracks()):
    print_particles(t, f"Track {i}")
    print(f"  Existence probability: {t.existence_probability()}")

# Check normalization
for i, t in enumerate(tracker.get_tracks()):
    weights = [p.weight for p in t.particles()]
    print(f"Track {i} weights sum: {sum(weights)}")
    print(f"Track {i} unique particles: {len(set(tuple(p.state_vector) for p in t.particles()))}")
