import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json

# Ensure lmb_engine is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lmb_engine', 'Debug')))
import lmb_engine

# Global configuration
DT = 60.0  # seconds
NUM_STEPS = 100
NUM_MONTE_CARLO_RUNS = 1
OSPA_CUTOFF = 1000.0  # meters

# Monte Carlo configurations to test
CONFIGURATIONS = [
    {'name': 'EKF-LMB Filter', 'initial_covariance_scale': 1.0},
    {'name': 'EKF-LMB Filter (High Cov)', 'initial_covariance_scale': 2.0},
]

def calculate_ospa_in_python(track_means, ground_truths, cutoff):
    """Calculate OSPA distance in Python to avoid needing to modify C++ code."""
    m = len(track_means)
    n = len(ground_truths)
    
    if m == 0 and n == 0:
        return 0.0  # No error
    if m == 0 or n == 0:
        return cutoff  # Maximum error
        
    # Determine which set is smaller for the distance matrix
    tracks_are_smaller = m <= n
    rows = min(m, n)
    cols = max(m, n)
    
    # Create distance matrix
    dist_matrix = np.zeros((rows, cols))
    
    if tracks_are_smaller:
        for i in range(m):
            for j in range(n):
                dist_matrix[i, j] = min(np.linalg.norm(track_means[i] - ground_truths[j]), cutoff)
    else:
        for i in range(n):
            for j in range(m):
                dist_matrix[i, j] = min(np.linalg.norm(track_means[j] - ground_truths[i]), cutoff)
    
    # Convert to a cost matrix for the assignment problem
    cost_matrix = np.copy(dist_matrix)
    
    # Solve assignment using LMB engine's solver
    hypotheses = lmb_engine.solve_assignment(cost_matrix, 1)
    
    if not hypotheses:
        return cutoff  # No valid assignment
    
    # Calculate OSPA based on optimal assignment
    assignment_sum = 0.0
    assoc_vec = hypotheses[0].associations
    
    for i in range(rows):
        j = assoc_vec[i] if i < len(assoc_vec) else -1
        if j != -1 and j >= 0 and j < cols:
            assignment_sum += dist_matrix[i, j]
        else:
            assignment_sum += cutoff
    
    # Add cardinality error
    cardinality_error = cutoff * abs(m - n)
    
    # Final OSPA score
    ospa = (assignment_sum + cardinality_error) / max(m, n)
    return ospa

def run_single_simulation(config):
    """Run a single simulation with the specified configuration."""
    print(f"Running simulation with {config['name']} configuration")
    
    # Load pre-generated ground truth data
    with open('ground_truth_data.json', 'r') as f:
        ground_truth_data = json.load(f)
    
    # Initialize models for EKF-LMB filter
    # Create the propagator - TwoBodyPropagator no longer takes process noise in constructor
    propagator = lmb_engine.TwoBodyPropagator()
    
    # Create the sensor model
    sensor_model = lmb_engine.InOrbitSensorModel()
    
    # Initial covariance for Gaussian components
    # This is the covariance for each Gaussian component, not particle spread
    scale = config['initial_covariance_scale']
    birth_cov = np.diag([scale * 50**2] * 3 + [scale * 5.0**2] * 3)
    
    # Birth model still uses num_particles param, but creates a single Gaussian component
    birth_model = lmb_engine.AdaptiveBirthModel(1, 0.3, birth_cov)
    
    # Create tracker with survival probability of 0.99
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)
    
    # Store OSPA scores for each step
    ospa_list = []
    
    # Main simulation loop
    for step in range(NUM_STEPS):
        print(f"Step {step+1}/{NUM_STEPS}")
        
        # Load ground truth states for this step
        current_gt_states = ground_truth_data[step]
        # Convert to numpy arrays - use first 6 components (position and velocity)
        gt_states = [np.array(gt[:6]) for gt in current_gt_states]
        
        # Generate measurements without clutter
        measurements = []
        for gt_state in gt_states:
            meas = lmb_engine.Measurement()
            meas.timestamp_ = step * DT
            # Add noise to position and velocity
            noisy_pos = gt_state[:3] + np.random.normal(0, 10, 3)
            noisy_vel = gt_state[3:6] + np.random.normal(0, 1, 3)
            meas.value_ = np.concatenate([noisy_pos, noisy_vel])
            meas.covariance_ = np.diag([50**2] * 3 + [5.0**2] * 3)
            measurements.append(meas)
        
        # Process noise covariance for EKF prediction
        # This is passed directly to the predict method now
        process_noise_q = np.diag([10**2] * 3 + [1.0**2] * 3)
        
        # Run the tracker prediction step
        tracker.predict(DT)
        
        # Get tracks after prediction for debugging
        if step == 1:  # After the first predict step
            predicted_tracks = tracker.get_tracks()
            for i, track in enumerate(predicted_tracks):
                particles = track.particles()
                if particles:
                    # In EKF version, we directly access the mean and covariance
                    print(f"Track {i} Predicted Mean: {particles[0].mean}")
                    print(f"Track {i} Predicted Covariance Diagonal: {np.diag(particles[0].covariance)}")
        
        # Run the tracker update step
        tracker.update(measurements)
        
        # Get tracks after update
        tracks = tracker.get_tracks()
        print(f"    Number of tracks: {len(tracks)}")
        
        # Print track existence probabilities
        for idx, t in enumerate(tracks):
            print(f"      Track {idx} existence probability: {t.existence_probability()}")
        
        # Extract track mean states for OSPA calculation
        track_means = []
        for track in tracks:
            particles = track.particles()
            if particles:
                # In EKF version, each track has a single Gaussian component
                # Just take the mean from the first (and only) particle
                mean_state = particles[0].mean
                track_means.append(mean_state)
        
        # Print ground truth and estimated states
        gt_states_full = [np.array(gt) for gt in current_gt_states]
        print("    Ground truth states:")
        for i, gt in enumerate(gt_states_full):
            print(f"      GT {i}: {gt}")
        
        print("    Track means:")
        for i, track_mean in enumerate(track_means):
            print(f"      Track {i}: {track_mean}")
        
        # Calculate errors between track means and ground truth
        for i, track_mean in enumerate(track_means):
            errors = [np.linalg.norm(track_mean - gt[:6]) for gt in gt_states_full]
            if errors:
                min_error = np.min(errors)
                closest_gt = gt_states_full[np.argmin(errors)]
                print(f"    Track {i}: min error to GT = {min_error:.3f}, closest GT = {closest_gt}")
        
        # Calculate OSPA using Python implementation to avoid modifying C++
        t0 = time.time()
        ospa_gt_states = [np.array(gt[:6]) for gt in current_gt_states]
        if track_means and ospa_gt_states:
            ospa = calculate_ospa_in_python(track_means, ospa_gt_states, OSPA_CUTOFF)
        else:
            ospa = OSPA_CUTOFF
        t1 = time.time()
        print(f"  OSPA: {ospa:.3f} (calculation time: {t1-t0:.3f}s)")
        
        ospa_list.append(ospa)
    
    return ospa_list

def main():
    print("Running Monte Carlo simulation with EKF-LMB tracker")
    print(f"Number of Monte Carlo runs: {NUM_MONTE_CARLO_RUNS}")
    print(f"Number of time steps per run: {NUM_STEPS}")
    
    plt.figure(figsize=(10, 6))
    
    # Run simulations for each configuration
    for config in CONFIGURATIONS:
        all_ospa = []
        
        for run in range(NUM_MONTE_CARLO_RUNS):
            print(f"Starting Monte Carlo run {run+1}/{NUM_MONTE_CARLO_RUNS}")
            ospa_list = run_single_simulation(config)
            all_ospa.append(ospa_list)
        
        # Average the OSPA results across all runs
        avg_ospa = np.mean(all_ospa, axis=0)
        
        # Plot the results
        plt.plot(avg_ospa, label=config['name'])
    
    plt.xlabel('Time Step')
    plt.ylabel('Average OSPA Distance (m)')
    plt.title('Monte Carlo OSPA Performance of EKF-LMB Filter')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ekf_lmb_ospa_performance.png')
    plt.show()

if __name__ == "__main__":
    main()