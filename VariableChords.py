import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# ========================
#  1. Constants & Loading
# ========================
EARTH_RADIUS = 6.371e6  # meters

# Load the PREM gravity function from a pickle file.
if os.path.exists("earth_gravity_fit.pkl"):
    with open("earth_gravity_fit.pkl", "rb") as f:
        gravity_function = pickle.load(f)
    print("earth_gravity_fit.pkl loaded successfully.")
else:
    raise FileNotFoundError("❌ ERROR: earth_gravity_fit.pkl is missing. The simulation cannot run.")


def prem_gravity(r):
    """
    Returns the gravitational acceleration (m/s²)
    at a radial distance r from Earth's center.
    """
    return gravity_function(r)


# ========================
#  2. Arc Distance to Central Angle
# ========================
def get_central_angle_from_arc(arc_distance, R=EARTH_RADIUS):
    """
    Given an arc (surface) distance, compute the central angle in radians:
         arc_distance = R * θ  =>  θ = arc_distance / R
    """
    if arc_distance <= 0:
        raise ValueError("Arc distance must be positive.")
    if arc_distance > np.pi * R:
        raise ValueError("Arc distance cannot exceed half Earth's circumference.")
    return arc_distance / R


# ========================
#  3. Single-Pass Chord Simulation Functions
# ========================
def chord_acceleration(s, central_angle):
    """
    Compute the acceleration along the chord at position s.
    Here, s=0 is the chord's midpoint, and s varies from -s0 to +s0.

    The perpendicular distance from Earth's center to the chord is:
         d = R * cos(θ/2)
    The radial distance is:
         r = sqrt(d² + s²)
    The component of gravity along the chord is:
         a(s) = - sign(s) * g(r) * (d / r)
    which pulls the train toward the midpoint (s = 0).
    """
    d = EARTH_RADIUS * np.cos(central_angle / 2.0)
    r = np.sqrt(d ** 2 + s ** 2)
    g_r = prem_gravity(r)
    return -np.sign(s) * g_r * (d / r)


def rk4_step_chord(s, v, dt, central_angle):
    """
    One RK4 integration step for:
         ds/dt = v
         dv/dt = chord_acceleration(s)
    """
    a1 = chord_acceleration(s, central_angle)
    k1_s = v * dt
    k1_v = a1 * dt

    a2 = chord_acceleration(s + 0.5 * k1_s, central_angle)
    k2_s = (v + 0.5 * k1_v) * dt
    k2_v = a2 * dt

    a3 = chord_acceleration(s + 0.5 * k2_s, central_angle)
    k3_s = (v + 0.5 * k2_v) * dt
    k3_v = a3 * dt

    a4 = chord_acceleration(s + k3_s, central_angle)
    k4_s = (v + k3_v) * dt
    k4_v = a4 * dt

    s_new = s + (k1_s + 2 * k2_s + 2 * k3_s + k4_s) / 6.0
    v_new = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
    return s_new, v_new


def simulate_chord_single_pass(arc_distance, dt=1.0):
    """
    Simulate a gravity train along a chord defined by an arc distance.

    Steps:
      1. Compute the central angle from the arc distance.
      2. Calculate half-chord length s0 = R sin(θ/2).
      3. Set initial conditions at s = -s0 (left surface), v = 0.
      4. Integrate until s >= s0 (right surface) using RK4.

    Safety limits (max time and iteration count) are included to prevent infinite loops.

    Returns:
      total_time      : Total trip time in seconds.
      time_list       : List of time values.
      s_list          : List of positions along the chord.
      velocity_list   : List of velocities.
      accel_list      : List of accelerations.
    """
    # 1) Compute the central angle.
    central_angle = get_central_angle_from_arc(arc_distance)

    # 2) Calculate half-chord length (s0).
    s0 = EARTH_RADIUS * np.sin(central_angle / 2.0)

    # 3) Initial conditions.
    s = -s0  # starting at the left surface
    v = 0.0  # initial velocity is zero
    t = 0.0

    time_list = []
    s_list = []
    velocity_list = []
    accel_list = []

    # Safety limits:
    MAX_TIME = 7200  # maximum simulation time in seconds (2 hours)
    MAX_ITER = 1000000  # maximum number of iterations
    iter_count = 0

    while s < s0 and t < MAX_TIME and iter_count < MAX_ITER:
        time_list.append(t)
        s_list.append(s)
        velocity_list.append(v)

        a = chord_acceleration(s, central_angle)
        accel_list.append(a)

        s, v = rk4_step_chord(s, v, dt, central_angle)
        t += dt
        iter_count += 1

        # Clamp if overshooting the right surface.
        if s >= s0:
            s = s0
            v = 0.0
            break
    else:
        print("Simulation terminated due to reaching maximum time or iterations.")

    return t, time_list, s_list, velocity_list, accel_list


# ========================
#  4. Main: Set Arc Distance and Run Simulation
# ========================
if __name__ == "__main__":
    # Change the arc distance here (in kilometers).
    arc_distance_km = 10000.0  # e.g., 1000 km along Earth's surface
    arc_distance_m = arc_distance_km * 1000.0

    # Set the time step (dt in seconds). Using a larger dt speeds up the simulation.
    dt = 1.0

    # Run the simulation.
    total_time, time_vals, s_vals, vel_vals, acc_vals = simulate_chord_single_pass(arc_distance_m, dt=dt)

    print(f"Arc distance = {arc_distance_km} km")
    print(f"Total trip time = {total_time:.2f} s = {total_time / 60:.2f} min")

    # =====================
    # Plot Velocity vs. Time
    # =====================
    plt.figure(figsize=(10, 5))
    plt.plot(time_vals, vel_vals, label="Velocity (m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Velocity vs. Time (Arc = {arc_distance_km} km)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # =====================
    # Plot Acceleration vs. Time
    # =====================
    plt.figure(figsize=(10, 5))
    plt.plot(time_vals, acc_vals, 'r', label="Acceleration (m/s²)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(f"Acceleration vs. Time (Arc = {arc_distance_km} km)")
    plt.grid(True)
    plt.legend()
    plt.show()
