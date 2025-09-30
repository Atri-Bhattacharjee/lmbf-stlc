# File: python/run_linear_sim.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- This block helps Python find the compiled C++ module ---
# Add the build directory to the Python path to find the lmb_engine module.
# This is a more robust way than manually copying the .pyd file.
# It assumes you are running this script from the project's root directory.

# Add the python/lmb_engine/Debug directory to sys.path if it exists
engine_debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python', 'lmb_engine', 'Debug'))
if os.path.isdir(engine_debug_dir):
    sys.path.append(engine_debug_dir)

build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
# On Windows, the output can be in a 'Debug' or 'Release' subfolder
debug_build_dir = os.path.join(build_dir, 'Debug')
release_build_dir = os.path.join(build_dir, 'Release')

if os.path.isdir(debug_build_dir):
    sys.path.append(debug_build_dir)
elif os.path.isdir(release_build_dir):
    sys.path.append(release_build_dir)
else:
    # As a fallback, check the build directory itself
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
# --- End of module finding block ---


def create_initial_track(num_particles=200):
    """
    Helper function to create a single C++ Track object for testing.
    This track will have a cloud of particles representing its state.
    """
    # Create track label
    label = lmb_engine.TrackLabel()
    label.birth_time = 0
    label.index = 1
    particles = []
    for _ in range(num_particles):
        p = lmb_engine.Particle()
        start_pos = np.random.randn(3) * 2.0
        start_vel = np.array([10.0, 5.0, 0.0])
        bc = 0.01
        p.state_vector = np.concatenate([start_pos, start_vel, [bc]])
        p.weight = 1.0 / num_particles
        particles.append(p)
    # Use correct Track constructor
    track = lmb_engine.Track(label, 0.99, particles)
    return track

def plot_particles(ax, tracks, color, label):
    """
    Helper function to extract particle positions from a list of Track objects
    and plot them on a matplotlib axes.
    """
    all_x = []
    all_y = []
    if not tracks:
        return # Do nothing if there are no tracks

    for track in tracks:
        for p in track.particles:
            # Extract x (index 0) and y (index 1) from the state vector
            all_x.append(p.state_vector[0])
            all_y.append(p.state_vector[1])
            
    ax.scatter(all_x, all_y, s=10, alpha=0.4, label=label, color=color)

def main():
    """
    Main function to orchestrate the validation test.
    """
    print("--- Validation Point I: Visual Predict Test ---")
    
    # 1. Instantiate the C++ models in Python
    print("Step 1: Instantiating C++ models...")
    propagator = lmb_engine.LinearPropagator()
    sensor_model = lmb_engine.SimpleSensorModel()
    
    # Define a dummy covariance matrix for the birth model constructor
    # Variances: 10m pos, 5m/s vel, 0.1 bc
    initial_cov = np.diag([10.0**2, 10.0**2, 10.0**2, 5.0**2, 5.0**2, 5.0**2, 0.1**2])
    birth_model = lmb_engine.AdaptiveBirthModel(100, 0.05, initial_cov)

    # 2. Instantiate the main C++ tracker engine
    print("Step 2: Instantiating C++ SMC_LMB_Tracker...")
    tracker = lmb_engine.create_smc_lmb_tracker(0.99)

    # 3. Create initial state and set it in the tracker
    print("Step 3: Creating initial track and setting state in C++ engine...")
    initial_track = create_initial_track()
    tracker.set_tracks([initial_track])

    # 4. Get the "before" state for plotting
    tracks_before = tracker.get_tracks()
    print(f"State Before: {len(tracks_before)} track(s), {len(tracks_before[0].particles)} particles.")

    # 5. --- RUN THE PREDICT STEP ---
    dt = 1.0  # Simulate a time step of 1.0 second
    print(f"Step 5: Calling tracker.predict(dt={dt})...")
    tracker.predict(dt)

    # 6. Get the "after" state for plotting
    tracks_after = tracker.get_tracks()
    print(f"State After: {len(tracks_after)} track(s), {len(tracks_after[0].particles)} particles.")

    # 7. Visualize the results
    print("Step 7: Plotting results...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_particles(ax, tracks_before, 'blue', 'State Before Predict')
    plot_particles(ax, tracks_after, 'red', 'State After Predict')
    
    # Calculate and plot the mean position for clarity
    mean_before = np.mean([p.state_vector[:2] for p in tracks_before[0].particles], axis=0)
    mean_after = np.mean([p.state_vector[:2] for p in tracks_after[0].particles], axis=0)
    ax.scatter(mean_before[0], mean_before[1], s=100, color='cyan', edgecolors='black', label='Mean Before')
    ax.scatter(mean_after[0], mean_after[1], s=100, color='magenta', edgecolors='black', label='Mean After')
    
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Validation Point I: Visual Predict Test")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    plt.show()
    print("--- Test Complete ---")

if __name__ == "__main__":
    # Ensure you have the required Python packages
    try:
        import numpy
        import matplotlib
    except ImportError:
        print("Please install required Python packages: pip install numpy matplotlib")
        sys.exit(1)
        
    main()