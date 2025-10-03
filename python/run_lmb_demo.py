import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'lmb_engine', 'Debug')))
import lmb_engine

DT = 10.0  # seconds
NUM_STEPS = 50
NUM_PARTICLES = 100
OSPA_CUTOFF = 10000.0

# Initialize two ground truth objects (7D: x, y, z, vx, vy, vz, bstar)
gt_init_states = [
    np.array([7000e3, 0, 0, 0, 7.546e3, 0, 1e-5]),
    np.array([7000e3, 100e3, 0, 0, 7.546e3, 0, 1e-5]),
]

# Sensor trajectory (6D: x, y, z, vx, vy, vz)
sensor_traj = [
    np.array([7050e3, 0, 0, 0, 7.546e3, 0]),
    np.array([7050e3, 100e3, 0, 0, 7.546e3, 0]),
]

# SGP4 process noise covariance (7x7)
pos_var = 0.1**2
vel_var = 0.01**2
bstar_var = 1e-8**2
sgp4_cov = np.diag([pos_var]*3 + [vel_var]*3 + [bstar_var])
propagator = lmb_engine.SGP4Propagator(sgp4_cov)
sensor_model = lmb_engine.InOrbitSensorModel()
birth_cov = np.diag([1e2**2]*3 + [1.0**2]*3 + [1e-4**2])
birth_model = lmb_engine.AdaptiveBirthModel(NUM_PARTICLES, 0.1, birth_cov)
tracker = lmb_engine.SMC_LMB_Tracker(propagator, sensor_model, birth_model, 0.99)

# Ground truth propagator (minimal noise, same as multistep test)
gt_process_noise_cov = np.diag([0.1]*3 + [0.01]*3 + [1e-8])
gt_propagator = lmb_engine.SGP4Propagator(gt_process_noise_cov)
gt_particles = []
for state in gt_init_states:
    p = lmb_engine.Particle()
    p.state_vector = state.copy()
    gt_particles.append(p)

ospa_list = []
for step in range(NUM_STEPS):
    # Propagate ground truths using SGP4 multistep logic
    for i in range(len(gt_particles)):
        gt_particles[i] = gt_propagator.propagate(gt_particles[i], DT, step * DT)
    # Propagate sensor trajectory
    sensor_state = sensor_traj[step % len(sensor_traj)]
    # Generate measurements
    measurements = []
    for gt in gt_particles:
        meas = lmb_engine.Measurement()
        meas.timestamp_ = step * DT
        noisy_pos = gt.state_vector[:3] + np.random.normal(0, 5, 3)
        noisy_vel = gt.state_vector[3:6] + np.random.normal(0, 0.05, 3)
        meas.value_ = np.concatenate([noisy_pos, noisy_vel])
        meas.covariance_ = np.diag([5**2]*3 + [0.05**2]*3)
        meas.sensor_id_ = "sensor"
        meas.sensor_state_ = sensor_state.copy()
        measurements.append(meas)
    tracker.predict(DT)
    tracker.update(measurements)
    tracks = tracker.get_tracks()
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
    gt_states_full = [p.state_vector[:6].copy() for p in gt_particles]
    if tracks and gt_states_full:
        ospa = lmb_engine.calculate_ospa_distance(tracks, gt_states_full, OSPA_CUTOFF)
    else:
        ospa = OSPA_CUTOFF
    ospa_list.append(ospa)
    print(f"Step {step+1}: OSPA={ospa:.2f}")
    print("  Ground truths:")
    for i, gt in enumerate(gt_particles):
        print(f"    GT {i}: {gt.state_vector}")
    print("  Track means:")
    for i, tm in enumerate(track_means):
        print(f"    Track {i}: {tm}")

plt.figure(figsize=(10, 6))
plt.plot(ospa_list, label="OSPA Distance")
plt.xlabel('Time Step')
plt.ylabel('OSPA Distance (m)')
plt.title('LMB Filter OSPA Performance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
