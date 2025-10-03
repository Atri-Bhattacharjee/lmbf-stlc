import numpy as np
import json
import sys
import os
# --- Robust module finding block ---
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
debug_build_dir = os.path.join(build_dir, 'Debug')
release_build_dir = os.path.join(build_dir, 'Release')
engine_debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'lmb_engine', 'Debug'))
if os.path.isdir(engine_debug_dir):
    sys.path.append(engine_debug_dir)
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
    sys.exit(1)

# Simulation parameters
N_STEPS = 20
DT = 60.0  # seconds
OUTPUT_FILE = 'ground_truth_data.json'

# Initial states for ground truth objects
initial_states = [
    # Ground Truth Object 1
    np.array([7000e3, 0.0, 0.0, 0.0, 7.546e3, 10.0]),
    # Ground Truth Object 2
    np.array([7000.2e3, 0.0, 0.0, 0.0, 7.546e3, -10.0])
]

# Zero process noise for deterministic ground truth
process_noise_cov = np.diag([0.0]*6)
gt_propagator = lmb_engine.TwoBodyPropagator(process_noise_cov)

gt_particles = []
for state in initial_states:
    p = lmb_engine.Particle()
    p.state_vector = state.copy()
    gt_particles.append(p)

all_ground_truths = []
print("Starting ground truth generation...")
for step in range(N_STEPS):
    # Store current 7D states for all ground truth objects
    current_states = [p.state_vector.tolist() for p in gt_particles]
    all_ground_truths.append(current_states)
    # Propagate each ground truth particle
    for i in range(len(gt_particles)):
        gt_particles[i] = gt_propagator.propagate(gt_particles[i], DT, step * DT)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_ground_truths, f, indent=4)
print(f"Ground truth generation complete. Data saved to {OUTPUT_FILE}.")
