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

def main():
    # ISS (ZARYA) TLE from Celestrak, 2025-09-30
    tle_line1 = "1 25544U 98067A   25273.59097222  .00002182  00000-0  48637-4 0  9992"
    tle_line2 = "2 25544  51.6412  36.2342 0004257  36.2342 323.7658 15.50000000  9992"
    print("Testing SGP4Propagator with ISS TLE:")
    print("TLE Line 1:", tle_line1)
    print("TLE Line 2:", tle_line2)
    # Create particle
    particle = lmb_engine.Particle()
    particle.tle_line1 = tle_line1
    particle.tle_line2 = tle_line2
    # Small process noise for deterministic output
    process_noise_cov = np.diag([1e-9]*6)
    propagator = lmb_engine.SGP4Propagator(process_noise_cov)
    dt_seconds = 3600.0  # 1 hour
    print(f"Propagating forward by {dt_seconds} seconds...")
    result_particle = propagator.propagate(particle, dt_seconds)
    state = result_particle.cartesian_state_vector
    pos = state[:3]
    vel = state[3:]
    print("\nResulting ECI State Vector:")
    print(f"Position (km): X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
    print(f"Velocity (km/s): VX={vel[0]:.6f}, VY={vel[1]:.6f}, VZ={vel[2]:.6f}")
    print("\nVerification Instructions:")
    print("1. Go to https://celestrak.org/js-utilities/TleProp.html")
    print("2. Paste the above TLE lines into the input box.")
    print(f"3. Set the propagation time to {dt_seconds} seconds.")
    print("4. Click 'Propagate' and note the ECI position vector.")
    print("5. Compare the ECI position from the website to the one printed above.")
    print("6. A close match (within a few km) confirms the test has passed.")

if __name__ == "__main__":
    main()
