import numpy as np
import matplotlib.pyplot as plt
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

# --- Simulation parameters ---
DT = 1.0
NUM_STEPS = 25
MEASUREMENT_NOISE_STD = 2.0  # meters
CLUTTER_RATE = 0
PLOT_BOUNDS = [-50, 50, -50, 50]  # xmin, xmax, ymin, ymax

# --- Ground truth objects ---
ground_truth = [
    {'position': np.array([-30.0, 20.0, 0.0]), 'velocity': np.array([2.0, -1.0, 0.0])},
    {'position': np.array([30.0, -30.0, 0.0]), 'velocity': np.array([-1.0, 2.0, 0.0])}
]

# --- Helper functions ---
def plot_particles(ax, tracks, color, label):
    all_x = []
    all_y = []
    if not tracks:
        return
    for track in tracks:
        for p in track.particles():
            all_x.append(p.state_vector[0])
            all_y.append(p.state_vector[1])
    ax.scatter(all_x, all_y, s=10, alpha=0.4, label=label, color=color)

def plot_ground_truth(ax, truth_objects):
    for obj in truth_objects:
        ax.scatter(obj['position'][0], obj['position'][1], s=120, marker='x', color='black', label='Ground Truth')

def plot_measurements(ax, measurements):
    mx = [m.value_[0] for m in measurements]
    my = [m.value_[1] for m in measurements]
    ax.scatter(mx, my, s=30, marker='+', color='gray', label='Measurements')

# --- Main simulation loop ---
def main():
    propagator = lmb_engine.LinearPropagator()
    sensor_model = lmb_engine.SimpleSensorModel()
    initial_cov = np.diag([MEASUREMENT_NOISE_STD**2]*3 + [5.0**2, 5.0**2, 5.0**2, 0.1**2])
    birth_model = lmb_engine.AdaptiveBirthModel(100, 0.05, initial_cov)
    tracker = lmb_engine.create_smc_lmb_tracker(0.99)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    for step in range(NUM_STEPS):
        ax.cla()
        # Advance ground truth
        for obj in ground_truth:
            obj['position'] += obj['velocity'] * DT
        # Generate measurements
        measurements = []
        for obj in ground_truth:
            meas = lmb_engine.Measurement()
            noisy_pos = obj['position'] + np.random.randn(3) * MEASUREMENT_NOISE_STD
            meas.value_ = noisy_pos
            meas.covariance_ = np.diag([MEASUREMENT_NOISE_STD**2]*3)
            meas.timestamp_ = step * DT
            meas.sensor_id_ = 'sim_sensor'
            measurements.append(meas)
        # Add clutter
        for _ in range(CLUTTER_RATE):
            clutter = lmb_engine.Measurement()
            clutter.value_ = np.array([
                np.random.uniform(PLOT_BOUNDS[0], PLOT_BOUNDS[1]),
                np.random.uniform(PLOT_BOUNDS[2], PLOT_BOUNDS[3]),
                0.0
            ])
            clutter.covariance_ = np.diag([MEASUREMENT_NOISE_STD**2]*3)
            clutter.timestamp_ = step * DT
            clutter.sensor_id_ = 'clutter'
            measurements.append(clutter)
        # Run filter
        tracker.predict(DT)
        tracker.update(measurements)
        tracks = tracker.get_tracks()
        confirmed_tracks = [tr for tr in tracks if tr.existence_probability() > 0.5]
        # Plot
        plot_ground_truth(ax, ground_truth)
        plot_measurements(ax, measurements)
        plot_particles(ax, tracks, 'blue', 'All Tracks')
        # Plot mean of confirmed tracks
        if confirmed_tracks:
            means = [np.mean([p.state_vector[:2] for p in tr.particles()], axis=0) for tr in confirmed_tracks]
            for mean in means:
                ax.scatter(mean[0], mean[1], s=100, color='magenta', edgecolors='black', label='Confirmed Mean')
        ax.set_xlim(PLOT_BOUNDS[0], PLOT_BOUNDS[1])
        ax.set_ylim(PLOT_BOUNDS[2], PLOT_BOUNDS[3])
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'End-to-End SMC_LMB_Tracker Simulation - Step {step+1}/{NUM_STEPS}')
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        plt.pause(0.1)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
