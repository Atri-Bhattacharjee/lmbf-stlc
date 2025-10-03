import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json  # Added import

# Ensure lmb_engine is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lmb_engine', 'Debug')))
import lmb_engine

DT = 60.0  # seconds
NUM_STEPS = 20
NUM_MONTE_CARLO_RUNS = 1
OSPA_CUTOFF = 10000.0  # meters

CONFIGURATIONS = [
    {'num_particles': 50, 'clutter_rate': 0},
]

def run_single_simulation(config):
    # Load pre-generated ground truth data
    with open('ground_truth_data.json', 'r') as f:
        ground_truth_data = json.load(f)

    # twobody process noise covariance (7x7)
    pos_var = 1**2
    vel_var = 0.1**2
    bstar_var = 1e-8**2
    # twobody_cov = np.diag([0.0]*7)
    twobody_cov = np.diag([pos_var]*3 + [vel_var]*3 + [bstar_var])
    propagator = lmb_engine.TwoBodyPropagator(twobody_cov)
    sensor_model = lmb_engine.InOrbitSensorModel()
    birth_cov = np.diag([10**2]*3 + [1.0**2]*3 + [1e-6**2])
    birth_model = lmb_engine.AdaptiveBirthModel(config['num_particles'], 0.9, birth_cov)
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)

    ospa_list = []
    for step in range(NUM_STEPS):
        print(f"Step {step+1}/{NUM_STEPS}")
        # Load ground truth states for this step
        current_gt_states = ground_truth_data[step]
        # Convert to numpy arrays for downstream use
        gt_states = [np.array(gt[:7]) for gt in current_gt_states]
        # Generate measurements (with clutter)
        measurements = []
        for gt_state in gt_states:
            meas = lmb_engine.Measurement()
            meas.timestamp_ = step * DT
            noisy_pos = gt_state[:3] + np.random.normal(0, 5, 3)
            noisy_vel = gt_state[3:6] + np.random.normal(0, 0.05, 3)
            noisy_bstar = gt_state[6] + np.random.normal(0, 1e-5)
            # noisy_pos = gt_state[:3] + np.random.normal(0, 0, 3)
            # noisy_vel = gt_state[3:6] + np.random.normal(0, 0, 3)
            # noisy_bstar = gt_state[6] + np.random.normal(0, 0)
            meas.value_ = np.concatenate([noisy_pos, noisy_vel, [noisy_bstar]])
            # meas.covariance_ = np.diag([1e-12]*7)
            meas.covariance_ = np.diag([500**2]*3 + [50**2]*3 + [1e-5**2])
            # meas.sensor_id_ = "0"
            # meas.sensor_state_ = gt_state[:6]
            measurements.append(meas)
        # num_clutter = np.random.poisson(config['clutter_rate'])
        # for _ in range(num_clutter):
        #     clutter_meas = lmb_engine.Measurement()
        #     clutter_meas.timestamp_ = step * DT
        #     clutter_pos = np.random.uniform(6900e3, 7100e3, 3)
        #     clutter_vel = np.random.uniform(-1e3, 1e3, 3)
        #     clutter_bstar = np.random.uniform(0, 2e-4)
        #     clutter_meas.value_ = np.concatenate([clutter_pos, clutter_vel, [clutter_bstar]])
        #     clutter_meas.covariance_ = np.diag([50**2]*3 + [0.5**2]*3 + [1e-4**2])
        #     clutter_meas.sensor_id_ = "0"
        #     clutter_meas.sensor_state_ = np.concatenate([clutter_pos, clutter_vel])
        #     measurements.append(clutter_meas)
        tracker.predict(DT)
        tracks = tracker.get_tracks()
        if step == 1: # After the first predict step
            predicted_tracks = tracker.get_tracks()
            for i, track in enumerate(predicted_tracks):
                particles = track.particles()
                if particles:
                    states = np.array([p.state_vector for p in particles])
                    cov = np.cov(states, rowvar=False)
                    # We only care about the diagonal (variances)
                    print(f"Track {i} Predicted Variances (pos/vel): {np.diag(cov)[:6]}")
        tracker.update(measurements)
        tracks = tracker.get_tracks()
        print(f"    Number of tracks: {len(tracks)}")
        for idx, t in enumerate(tracks):
            print(f"      Track {idx} existence probability: {t.existence_probability()}")
        track_means = []
        for t in tracks:
            if hasattr(t, 'particles'):
                ps = t.particles()
                if ps:
                    mean = np.zeros(7)
                    total_weight = 0.0
                    for p in ps:
                        mean += p.state_vector * p.weight
                        total_weight += p.weight
                    if total_weight > 0.0:
                        mean /= total_weight
                    track_means.append(mean)
        gt_states_full = [np.array(gt) for gt in current_gt_states]
        # Print ground truth states
        print("    Ground truth states:")
        for i, gt in enumerate(gt_states_full):
            print(f"      GT {i}: {gt}")
        # Print track means
        print("    Track means:")
        for i, track_mean in enumerate(track_means):
            print(f"      Track {i}: {track_mean}")
        for i, track_mean in enumerate(track_means):
            errors = [np.linalg.norm(track_mean - gt) for gt in gt_states_full]
            min_error = np.min(errors)
            closest_gt = gt_states_full[np.argmin(errors)]
            print(f"    Track {i}: min error to GT = {min_error:.3f}, track state = {track_mean}, closest GT = {closest_gt}")
        t0 = time.time()
        ospa_gt_states_full = [np.array(gt[:6]) for gt in current_gt_states]
        if tracks and ospa_gt_states_full:
            ospa = lmb_engine.calculate_ospa_distance(tracks, ospa_gt_states_full, OSPA_CUTOFF)
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
