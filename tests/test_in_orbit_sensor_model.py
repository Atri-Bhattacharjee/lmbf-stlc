import sys
import os
import numpy as np

# Boilerplate to find lmb_engine in correct output directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python/lmb_engine/Debug')))
import lmb_engine

def main():
    # 1. Test data setup
    sensor_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    debris_state = np.array([7010.0, 0.0, 0.0, 0.0, 7.5, 0.0])

    # 2. Manual ground truth calculation
    relative_pos = debris_state[:3] - sensor_state[:3]
    gt_range = np.linalg.norm(relative_pos)
    gt_azimuth = np.arctan2(relative_pos[1], relative_pos[0])
    gt_elevation = np.arcsin(relative_pos[2] / gt_range)
    gt_measurement_value = np.array([gt_azimuth, gt_elevation, gt_range])

    # 3. Instantiate C++ objects
    sensor_model = lmb_engine.InOrbitSensorModel()
    particle = lmb_engine.Particle()
    particle.cartesian_state_vector = debris_state
    measurement = lmb_engine.Measurement()
    measurement.sensor_state_ = sensor_state
    measurement.value_ = gt_measurement_value
    measurement.covariance_ = np.eye(3) * 1e-9

    # 4. Execute and validate
    likelihood = sensor_model.calculate_likelihood(particle, measurement)
    print("Sensor ECI state:", sensor_state)
    print("Debris ECI state:", debris_state)
    print("Ground truth measurement value:", gt_measurement_value)
    print("Likelihood:", likelihood)
    assert abs(likelihood - 1.0) < 1e-6, f"Expected likelihood ~1.0, got {likelihood}"
    print("Test Passed!")

if __name__ == "__main__":
    main()
