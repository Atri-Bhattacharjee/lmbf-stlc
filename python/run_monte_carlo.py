import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json  # Added import
import random

# Ensure lmb_engine is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lmb_engine', 'Debug')))
import lmb_engine

DT = 60.0  # seconds
NUM_STEPS = 50
NUM_MONTE_CARLO_RUNS = 3  # Each super run consists of 3 Monte Carlo runs
NUM_SUPER_RUNS = 20  # Number of super runs to perform and average
OSPA_CUTOFF = 100000.0  # meters

CONFIGURATIONS = [
    {'num_particles': 100, 'clutter_rate': 0},  # Clutter rate must be 0 for the simplified tracker
]

def run_single_simulation(config, seed=None):
    # Set the random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Load pre-generated ground truth data
    with open('ground_truth_data.json', 'r') as f:
        ground_truth_data = json.load(f)

    # twobody process noise covariance (6x6)
    pos_var = 10**2
    vel_var = 1.0**2
    # twobody_cov = np.diag([0.0]*6)
    twobody_cov = np.diag([pos_var]*3 + [vel_var]*3)
    propagator = lmb_engine.TwoBodyPropagator(twobody_cov)
    sensor_model = lmb_engine.InOrbitSensorModel()
    birth_cov = np.diag([50**2]*3 + [5.0**2]*3)
    birth_model = lmb_engine.AdaptiveBirthModel(config['num_particles'], 0.3, birth_cov)
    tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)

    # Data collection for performance analysis
    ospa_list = []
    track_errors = []  # Will store minimum errors for each track at each time step
    state_component_errors = []  # Will store 6-element error vectors for best-matched tracks
    for step in range(NUM_STEPS):
        print(f"Step {step+1}/{NUM_STEPS}")
        # Load ground truth states for this step
        current_gt_states = ground_truth_data[step]
        # Convert to numpy arrays for downstream use
        gt_states = [np.array(gt[:6]) for gt in current_gt_states]
        # Generate measurements (without clutter to match simplified tracker assumptions)
        # The simplified tracker requires num_tracks == num_measurements
        measurements = []
        for gt_state in gt_states:
            meas = lmb_engine.Measurement()
            meas.timestamp_ = step * DT
            noisy_pos = gt_state[:3] + np.random.normal(0, 1000 / (step + 1), 3)
            noisy_vel = gt_state[3:6] + np.random.normal(0, 100 / (step + 1), 3)
            meas.value_ = np.concatenate([noisy_pos, noisy_vel])
            meas.covariance_ = np.diag([(1000 / (step + 1))**2]*3 + [(100 / (step + 1))**2]*3)
            measurements.append(meas)
        # NOTE: Clutter generation is disabled for the simplified tracker
        # The new tracker implementation requires num_tracks == num_measurements
        # and doesn't handle missed detections or false alarms
        #
        # num_clutter = np.random.poisson(config['clutter_rate'])
        # for _ in range(num_clutter):
        #     clutter_meas = lmb_engine.Measurement()
        #     clutter_meas.timestamp_ = step * DT
        #     clutter_pos = np.random.uniform(6900e3, 7100e3, 3)
        #     clutter_vel = np.random.uniform(-1e3, 1e3, 3)
        #     clutter_meas.value_ = np.concatenate([clutter_pos, clutter_vel])
        #     clutter_meas.covariance_ = np.diag([50**2]*3 + [0.5**2]*3)
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
                    mean = np.zeros(6)
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
        # Collect track-to-truth error metrics
        step_track_errors = []
        step_component_errors = None  # Initialize as None instead of empty list
        
        for i, track_mean in enumerate(track_means):
            errors = [np.linalg.norm(track_mean - gt) for gt in gt_states_full]
            min_error = np.min(errors)
            min_error_idx = np.argmin(errors)
            closest_gt = gt_states_full[min_error_idx]
            print(f"    Track {i}: min error to GT = {min_error:.3f}, track state = {track_mean}, closest GT = {closest_gt}")
            
            # Store the minimum error for this track
            step_track_errors.append(min_error)
            
            # For the first track, calculate component-wise errors
            if i == 0 and len(gt_states_full) > 0:
                # Calculate error for each state component
                error_vec = track_mean[:6] - closest_gt[:6]
                step_component_errors = error_vec
        
        # Store track errors for this step
        track_errors.append(step_track_errors)
        
        # Store component errors if we have them
        if step_component_errors is not None:
            state_component_errors.append(step_component_errors)
        
        t0 = time.time()
        ospa_gt_states_full = [np.array(gt[:6]) for gt in current_gt_states]
        if tracks and ospa_gt_states_full:
            ospa = lmb_engine.calculate_ospa_distance(tracks, ospa_gt_states_full, OSPA_CUTOFF)
        else:
            ospa = OSPA_CUTOFF
        t1 = time.time()
        print(f"  ospa: {t1-t0:.3f}s")
        ospa_list.append(ospa)
    return ospa_list, track_errors, state_component_errors

def main():
    print("Running Monte Carlo simulation with simplified tracker (no missed detections)")
    print("Using one-to-one assignment model where num_tracks must equal num_measurements")
    print(f"Performing {NUM_SUPER_RUNS} super runs, each with {NUM_MONTE_CARLO_RUNS} offset Monte Carlo simulations")
    
    # Data collection across all configurations and runs
    for config in CONFIGURATIONS:
        # List to store combined OSPA results from each super run
        all_super_run_ospa = []
        
        # Perform multiple super runs
        for super_run in range(NUM_SUPER_RUNS):
            print(f"\nStarting Super Run {super_run+1}/{NUM_SUPER_RUNS}")
            
            all_ospa = []
            all_track_errors = []
            all_component_errors = []
            
            for run in range(NUM_MONTE_CARLO_RUNS):
                # Create a unique seed based on super_run and run indices
                # This ensures every Monte Carlo simulation gets a different random sequence
                unique_seed = super_run * 1000 + run * 100 + int(time.time() % 100)
                print(f"  Running Monte Carlo simulation {run+1}/{NUM_MONTE_CARLO_RUNS} of Super Run {super_run+1} (seed={unique_seed})")
                ospa_list, track_errors, state_component_errors = run_single_simulation(config, seed=unique_seed)
                all_ospa.append(ospa_list)
                all_track_errors.append(track_errors)
                all_component_errors.append(state_component_errors)
            
            # Process this super run's data with offsets
            # Determine the maximum time step after applying offsets
            max_time_step = NUM_STEPS
            offsets = [0, 30, 50]  # Offsets for 1st, 2nd, and 3rd runs
            
            # Create a standard time axis for non-offset plots (track errors and component errors)
            track_time_axis = np.arange(1, NUM_STEPS)
            
            # Only apply offsets if we have multiple runs
            if NUM_MONTE_CARLO_RUNS > 1:
                # Calculate the maximum length needed for the offset data
                for i in range(min(NUM_MONTE_CARLO_RUNS, len(offsets))):
                    if i > 0:  # Skip the first run (no offset)
                        max_time_step = max(max_time_step, NUM_STEPS + offsets[i])
                
                # Create a combined array with offsets
                combined_ospa = np.full(max_time_step, np.nan)
                count_array = np.zeros(max_time_step)
                
                # Add each run with its appropriate offset
                for i in range(min(NUM_MONTE_CARLO_RUNS, len(offsets))):
                    ospa_run = all_ospa[i]
                    offset = offsets[i]
                    
                    # Add this run's data to the combined array with the specified offset
                    for t in range(len(ospa_run)):
                        idx = t + offset
                        if idx < max_time_step:
                            if np.isnan(combined_ospa[idx]):
                                combined_ospa[idx] = ospa_run[t]
                            else:
                                combined_ospa[idx] += ospa_run[t]
                            count_array[idx] += 1
                
                # Calculate the average (avoiding division by zero)
                for i in range(max_time_step):
                    if count_array[i] > 0:
                        combined_ospa[i] /= count_array[i]
                
                # Create time axis starting at step 1 (exclude step 0) up to max_time_step
                time_axis = np.arange(1, max_time_step)
                
                # Store this super run's combined OSPA data for averaging later
                all_super_run_ospa.append(combined_ospa[1:])
                
                # Create figure for individual super run OSPA plot if desired
                if NUM_SUPER_RUNS <= 3:  # Only show individual super run plots if we have 3 or fewer
                    plt.figure(figsize=(10, 6))
                    plt.plot(time_axis, combined_ospa[1:], label=f"Super Run {super_run+1}")
                    plt.xlabel('Time Step')
                    plt.ylabel('Combined OSPA Distance (m)')
                    plt.title(f'Super Run {super_run+1}: OSPA Performance with Offset Object Introduction (at t=30 and t=50)')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
        else:
            # If only one run, just use the original data without offsets
            # Convert to numpy arrays for easier manipulation
            avg_ospa = np.array(all_ospa[0])
            
            # Create time axis starting at step 1 (exclude step 0)
            time_axis = np.arange(1, NUM_STEPS)
            
            # Create figure 1: OSPA plot (excluding step 0)
            plt.figure(figsize=(10, 6))
            plt.plot(time_axis, avg_ospa[1:], label=f"Particles={config['num_particles']}")
            plt.xlabel('Time Step')
            plt.ylabel('OSPA Distance (m)')
            plt.title('Monte Carlo OSPA Performance (Single Run)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        
        # Create a final plot that averages all the super runs together
        if NUM_SUPER_RUNS > 1 and len(all_super_run_ospa) > 0:
            # Find the minimum length of all super run results
            min_length = min(len(ospa_data) for ospa_data in all_super_run_ospa)
            
            # Create a new array to hold the average
            super_run_avg_ospa = np.zeros(min_length)
            
            # Sum all super run results
            for ospa_data in all_super_run_ospa:
                super_run_avg_ospa += ospa_data[:min_length]
            
            # Divide by number of super runs to get average
            super_run_avg_ospa /= len(all_super_run_ospa)
            
            # Create the time axis for the averaged data
            avg_time_axis = np.arange(1, min_length+1)
            
            # Create a plot with all super runs shown as light thin lines
            plt.figure(figsize=(10, 6))
            
            # Plot each individual super run with light, thin, translucent lines
            individual_line_added_to_legend = False
            for i, ospa_data in enumerate(all_super_run_ospa):
                # For the first super run, add to legend; for others, don't add to avoid legend clutter
                if not individual_line_added_to_legend:
                    plt.plot(avg_time_axis, ospa_data[:min_length], 'c-', linewidth=0.8, alpha=0.3, 
                             label="Individual Super Runs")
                    individual_line_added_to_legend = True
                else:
                    plt.plot(avg_time_axis, ospa_data[:min_length], 'c-', linewidth=0.8, alpha=0.3)
            
            # Average line removed as requested
            
            plt.xlabel('Time Step')
            plt.ylabel('OSPA Distance (m)')
            plt.title(f'OSPA Distance Plot of {NUM_SUPER_RUNS} Runs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Create the average-only super run plot
            plt.figure(figsize=(10, 6))
            plt.plot(avg_time_axis, super_run_avg_ospa, 'k-', linewidth=2, label=f"Average of {NUM_SUPER_RUNS} Super Runs")
            plt.xlabel('Time Step')
            plt.ylabel('Average OSPA Distance (m)')
            plt.title(f'Average OSPA Performance Across {NUM_SUPER_RUNS} Runs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        
        # Process track errors
        # Convert to numpy array and handle potential variable number of tracks per step
        # First find the maximum number of tracks in any step
        max_tracks = 0
        for run_errors in all_track_errors:
            for step_errors in run_errors:
                max_tracks = max(max_tracks, len(step_errors))
        
        # Initialize array for average track errors
        avg_track_errors = np.full((NUM_STEPS, max_tracks), np.nan)
        
        # Fill in the array with available track errors
        for run_idx, run_errors in enumerate(all_track_errors):
            for step_idx, step_errors in enumerate(run_errors):
                for track_idx, error in enumerate(step_errors):
                    if np.isnan(avg_track_errors[step_idx, track_idx]):
                        avg_track_errors[step_idx, track_idx] = error
                    else:
                        avg_track_errors[step_idx, track_idx] += error
        
        # Compute average by dividing by number of runs
        avg_track_errors /= NUM_MONTE_CARLO_RUNS
        
        # Figure 3: Per-track minimum error
        plt.figure(figsize=(10, 6))
        
        for track_idx in range(min(max_tracks, 5)):  # Show up to 5 tracks to avoid clutter
            # Skip if all values are NaN (track doesn't exist throughout the simulation)
            if not np.all(np.isnan(avg_track_errors[1:, track_idx])):
                plt.plot(track_time_axis, avg_track_errors[1:, track_idx])

        plt.xlabel('Time Step')
        plt.ylabel('Minimum Error Distance (m)')
        plt.title('Minimum Error Over Time')
        # Legend removed as requested
        plt.grid(True)
        plt.tight_layout()
        
        # Process component errors
        # Convert all_component_errors to numpy array (shape: num_runs x num_steps x 6)
        # Handle potential issues with missing data
        component_errors_array = []
        for run_errors in all_component_errors:
            # Check if run_errors is not empty
            if len(run_errors) > 0:
                component_errors_array.append(run_errors)
        
        if len(component_errors_array) > 0:
            avg_component_errors = np.mean(component_errors_array, axis=0)
            
            # Figure 2: State component errors (6-panel plot)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
            component_titles = ['X Error (m)', 'Y Error (m)', 'Z Error (m)', 
                            'Vx Error (m/s)', 'Vy Error (m/s)', 'Vz Error (m/s)']
            
            for i, title in enumerate(component_titles):
                row, col = i // 3, i % 3
                ax = axes[row, col]
                
                # Plot error for this component (skip step 0)
                if len(avg_component_errors) > 1:  # Make sure we have data beyond step 0
                    component_data = avg_component_errors[1:, i]
                    # Use track_time_axis for component errors as well to ensure dimensions match
                    ax.plot(track_time_axis, component_data)
                ax.set_title(title)
                ax.grid(True)
                
                # Add horizontal line at y=0 for reference
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                
                # Only add x-label to bottom row
                if row == 1:
                    ax.set_xlabel('Time Step')
                
                # Only add y-label to leftmost column
                if col == 0:
                    ax.set_ylabel('Error')
            
            plt.tight_layout()
    
    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
