import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Ensure lmb_engine is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lmb_engine', 'Debug')))
import lmb_engine

DT = 1.0  # seconds
NUM_STEPS = 20
NUM_MONTE_CARLO_RUNS = 1
OSPA_CUTOFF = 10000.0  # meters

CONFIGURATIONS = [
    {'num_particles': 50, 'clutter_rate': 0},
]

def run_single_simulation(config):
    # Initial ground truth ECI states (meters, m/s)
    gt_init_states = [
        np.array([7000e3, 0, 0, 0, 7.546e3, 0]),
        np.array([7000e3, 100e3, 0, 0, 7.546e3, 0]),
    ]
    # Covariance for birth model
    cov = np.eye(7) * 0.1
    tracker = lmb_engine.create_custom_smc_lmb_tracker(
        config['num_particles'], 0.1, cov, 0.99)
    # Create SGP4Propagator for ground truth propagation
    sgp4_cov = np.diag([1e-2]*3 + [1e-5]*3)
    gt_propagator = lmb_engine.SGP4Propagator(sgp4_cov)
    # Initial ground truth particles (with state_vector only)
    gt_particles = []
    for state in gt_init_states:
        p = lmb_engine.Particle()
        p.state_vector = np.concatenate([state, [0.0]]) # 7D: [pos, vel, ballistic coeff]
        gt_particles.append(p)
    ospa_list = []
    for step in range(NUM_STEPS):
        print(f"Step {step+1}/{NUM_STEPS}")
        # Propagate ground truths using SGP4
        for i in range(len(gt_particles)):
            gt_particles[i] = gt_propagator.propagate(gt_particles[i], DT)
        # Extract updated ground truth states and velocities from state_vector
        gt_states = [p.state_vector[:3] for p in gt_particles]
        gt_vels = [p.state_vector[3:6] for p in gt_particles]
        # Generate measurements (with clutter)
        measurements = []
        for gt_pos, gt_vel in zip(gt_states, gt_vels):
            meas = lmb_engine.Measurement()
            meas.timestamp_ = step * DT
            # Measurement is noisy ECI position and velocity
            noisy_pos = gt_pos + np.random.normal(0, 5, 3)
            noisy_vel = gt_vel + np.random.normal(0, 0.05, 3)
            meas.value_ = np.concatenate([noisy_pos, noisy_vel])
            meas.covariance_ = np.diag([5**2]*3 + [0.05**2]*3)
            meas.sensor_id_ = "0"
            meas.sensor_state_ = np.zeros(7)
            measurements.append(meas)
        # Add clutter
        num_clutter = np.random.poisson(config['clutter_rate'])
        for _ in range(num_clutter):
            clutter_meas = lmb_engine.Measurement()
            clutter_meas.timestamp_ = step * DT
            clutter_pos = np.random.uniform(6900e3, 7100e3, 3)
            clutter_vel = np.random.uniform(-1e3, 1e3, 3)
            clutter_meas.value_ = np.concatenate([clutter_pos, clutter_vel])
            clutter_meas.covariance_ = np.diag([50**2]*3 + [0.5**2]*3)
            clutter_meas.sensor_id_ = "0"
            clutter_meas.sensor_state_ = np.zeros(7)
            measurements.append(clutter_meas)
        tracker.predict(DT)
        # Print particle weights before update (before resampling)
        tracks = tracker.get_tracks()
        for idx, t in enumerate(tracks):
            if hasattr(t, 'particles'):
                ps = t.particles()
                if ps:
                    weights = [p.weight for p in ps]
                    print(f"      Track {idx} weights before update: {weights}")
        tracker.update(measurements)
        tracks = tracker.get_tracks()
        # Print particle weights after update (after resampling)
        for idx, t in enumerate(tracks):
            if hasattr(t, 'particles'):
                ps = t.particles()
                if ps:
                    weights = [p.weight for p in ps]
                    print(f"      Track {idx} weights after update: {weights}")
        print(f"    Number of tracks: {len(tracks)}")
        for idx, t in enumerate(tracks):
            print(f"      Track {idx} existence probability: {t.existence_probability()}")
        # Compute and print error between track means and closest ground truth
        track_means = []
        for t in tracks:
            if hasattr(t, 'particles'):
                ps = t.particles()
                if ps:
                    mean = np.zeros(6)
                    total_weight = 0.0
                    for p in ps:
                        mean[:6] += p.state_vector[:6] * p.weight
                        total_weight += p.weight
                    if total_weight > 0.0:
                        mean /= total_weight
                    track_means.append(mean)
        gt_states_full = [np.concatenate([gt_states[i], gt_vels[i]]) for i in range(len(gt_states))]
        for i, track_mean in enumerate(track_means):
            errors = [np.linalg.norm(track_mean - gt) for gt in gt_states_full]
            min_error = np.min(errors)
            closest_gt = gt_states_full[np.argmin(errors)]
            print(f"    Track {i}: min error to GT = {min_error:.3f}, track state = {track_mean}, closest GT = {closest_gt}")
        t0 = time.time()
        if tracks and gt_states_full:
            ospa = lmb_engine.calculate_ospa_distance(tracks, gt_states_full, OSPA_CUTOFF)
        else:
            ospa = OSPA_CUTOFF
        t1 = time.time()
        print(f"  ospa: {t1-t0:.3f}s")
        ospa_list.append(ospa)
    return ospa_list

def main():
    plt.figure(figsize=(10, 6))
    for config in CONFIGURATIONS:
        all_ospa = []
        for run in range(NUM_MONTE_CARLO_RUNS):
            ospa_list = run_single_simulation(config)
            all_ospa.append(ospa_list)
        avg_ospa = np.mean(all_ospa, axis=0)
        plt.plot(avg_ospa, label=f"Particles={config['num_particles']}, Clutter={config['clutter_rate']}")
    plt.xlabel('Time Step')
    plt.ylabel('Average OSPA Distance (m)')
    plt.title('Monte Carlo OSPA Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
