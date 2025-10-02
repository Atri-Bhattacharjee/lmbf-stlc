import numpy as np
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

# Test parameters
N_STEPS = 10
DT = 60.0  # seconds

# Initial state: circular LEO orbit (example)
state_vector = np.array([
    7000e3, 0.0, 0.0,   # position (m)
    0.0, 7.546e3, 0.0   # velocity (m/s)
])
particle = lmb_engine.Particle()
particle.state_vector = state_vector

# Small process noise (diagonal)
process_noise_cov = np.diag([1e-2]*3 + [1e-5]*3)
propagator = lmb_engine.SGP4Propagator(process_noise_cov)

print("Step | state_vector")
for step in range(N_STEPS):
    particle = propagator.propagate(particle, DT)
    print(f"{step:2d} | {particle.state_vector}")
