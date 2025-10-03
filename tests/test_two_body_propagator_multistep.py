import numpy as np
import os
import sys
sys.path.append('python/lmb_engine/Debug')
import lmb_engine

def test_two_body_multistep():
    print("Test: TwoBodyPropagator multistep propagation")
    # Initial state: [x, y, z, vx, vy, vz, bstar]
    state = np.array([7000e3, 0, 0, 0, 7.546e3, 0, 1e-4], dtype=float)
    process_noise = np.diag([0.0]*7)
    propagator = lmb_engine.TwoBodyPropagator(process_noise)
    p = lmb_engine.Particle()
    p.state_vector = state.copy()
    p.weight = 1.0
    dt = 60.0  # seconds
    steps = 20
    states = [p.state_vector.copy()]
    for i in range(steps):
        p = propagator.propagate(p, dt, i*dt)
        states.append(p.state_vector.copy())
    states = np.array(states)
    print("Propagated states (first 3 steps):")
    for i in range(3):
        print(f"Step {i}: {states[i]}")
    # Check energy conservation (should be nearly constant for two-body)
    mu = 3.986004418e14
    energies = []
    for s in states:
        r = np.linalg.norm(s[:3])
        v = np.linalg.norm(s[3:6])
        energy = 0.5*v**2 - mu/r
        energies.append(energy)
    energies = np.array(energies)
    print("Specific orbital energies:", energies)
    delta_energy = np.max(energies) - np.min(energies)
    print(f"Max energy change: {delta_energy}")
    assert delta_energy < 1e5, f"Energy drift too large: {delta_energy}"
    print("PASS: TwoBodyPropagator multistep test")

if __name__ == "__main__":
    test_two_body_multistep()
