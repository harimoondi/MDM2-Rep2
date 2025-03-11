import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Constants
EARTH_RADIUS = 6.371e6  # meters
SURFACE_GRAVITY = 9.807  # m/s²

# Load the gravity function from earth_gravity_fit.pkl
if os.path.exists("earth_gravity_fit.pkl"):
    with open("earth_gravity_fit.pkl", "rb") as f:
        gravity_function = pickle.load(f)
    print("earth_gravity_fit.pkl loaded successfully.")
else:
    raise FileNotFoundError("❌ ERROR: earth_gravity_fit.pkl is missing. The simulation cannot run.")

def prem_gravity(r):
    """Retrieve gravity from the precomputed function and ensure realistic values."""
    g_r = gravity_function(r)
    return g_r

def acceleration(r, tunnel_angle):
    """Compute acceleration along the tunnel's axis using PREM gravity."""
    g_r = prem_gravity(r)  # Get gravity from the PREM model
    a_tunnel = g_r * np.cos(tunnel_angle)  # Projected gravity along tunnel
    return -a_tunnel  # Negative since it pulls toward the center

def rk4_step(r, v, dt, tunnel_angle):
    """Perform one RK4 integration step with fixed dt."""
    adaptive_dt = dt  # Avoid adaptive scaling

    k1_v = acceleration(r, tunnel_angle) * adaptive_dt
    k1_r = v * adaptive_dt

    k2_v = acceleration(r + 0.5 * k1_r, tunnel_angle) * adaptive_dt
    k2_r = (v + 0.5 * k1_v) * adaptive_dt

    k3_v = acceleration(r + 0.5 * k2_r, tunnel_angle) * adaptive_dt
    k3_r = (v + 0.5 * k2_v) * adaptive_dt

    k4_v = acceleration(r + k3_r, tunnel_angle) * adaptive_dt
    k4_r = (v + k3_v) * adaptive_dt

    r_new = r + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    v_new = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return r_new, v_new

def simulate_halfway_gravity_train(dt=0.01):
    """Simulate from surface to Earth's center (half journey)."""
    r = EARTH_RADIUS  # Start at surface
    v = 0.0  # Initial velocity
    t = 0.0

    tunnel_ang = 0  # Assume tunnel goes through the center

    time_values = []
    position_values = []
    velocity_values = []
    acceleration_values = []

    # Move from surface to the center
    while r > 0:
        time_values.append(t)
        position_values.append(r)
        velocity_values.append(v)  # Store signed velocity
        acc = acceleration(r, tunnel_ang)
        acceleration_values.append(acc)

        # Print Debugging Data
        print(f"Time: {t:.2f} s, Velocity: {v:.2f} m/s, Acceleration: {acc:.2f} m/s²")

        r, v = rk4_step(r, v, dt, tunnel_ang)
        t += dt

        if r <= 0:
            r = 0  # Ensure no overshoot
            break

    # Double the time to estimate full one-way trip
    total_time_one_way = 2 * t
    return total_time_one_way, time_values, position_values, velocity_values, acceleration_values

# Run Simulation
full_time_one_way, time_values_half, position_values_half, velocity_values_half, acceleration_values_half = \
    simulate_halfway_gravity_train(dt=0.01)

print(f"RK4 Travel Time (One-Way Trip Estimated): {full_time_one_way / 60:.2f} minutes")

# Extend data for full one-way trip by mirroring the second half
time_values_full = time_values_half + [t + time_values_half[-1] for t in time_values_half]
position_values_full = position_values_half + position_values_half[::-1]
velocity_values_full = velocity_values_half + [-v for v in velocity_values_half[::-1]]  # Flip sign in second half
acceleration_values_full = acceleration_values_half + [-a for a in acceleration_values_half[::-1]]

# Generate and Save Corrected Velocity Graph
plt.figure(figsize=(10, 5))
plt.plot(time_values_full, velocity_values_full, label="Velocity (m/s)")
plt.xlabel("Time (seconds)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs. Time for Gravity Train (One-Way Trip)")
plt.legend()
plt.grid(True)

plt.text(max(time_values_full) * 0.6, min(velocity_values_full) * 0.6,
         f"Total Time: {full_time_one_way / 60:.2f} min", fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

plt.savefig("corrected_velocity_vs_time_one_way.png")  # Save the figure
plt.show()

# Generate and Save Acceleration Graph
plt.figure(figsize=(10, 5))
plt.plot(time_values_full, acceleration_values_full, label="Acceleration (m/s²)", color='red')
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration vs. Time for Gravity Train (One-Way Trip)")
plt.legend()
plt.grid(True)

plt.text(max(time_values_full) * 0.7, min(acceleration_values_full) * 0.4,
         f"Total Time: {full_time_one_way / 60:.2f} min", fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8))

plt.savefig("corrected_acceleration_vs_time_one_way.png")  # Save the figure
plt.show()
