import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.interpolate import CubicSpline

# Constants
R_earth = 6371000  # Earth radius in meters
G = const.G
M = 5.972e24
R = 6371000


# Linear gravitational
def linear_I(r):
    return (G * M / R**3) * r

def PREM_I(r):
    file_path = "Gravity and Density Functions/functions/earth_gravity_fit.pkl"
    with open(file_path, "rb") as f:
        non_gravity_function = pickle.load(f)
    return non_gravity_function(r)


def calculate_dr_dh(r, R_d):
    dr_dh = r / R_d * np.sqrt((r ** 2 * PREM_I(R_d) - R_d**2 * PREM_I(r) / PREM_I(r)))
    return dr_dh


def calculate_path(R_d, dh=0.001):
    """
    Calculate path through Earth with maximum depth R_d
    dh: step size for angle integration
    """
    # Load gravity model
    # Starting conditions (small offset from maximum depth point)
    r = R_d - 1.0  # Small offset to avoid r=0
    h = 0.0

    # Arrays to store path
    r_values = [r]
    h_values = [h]

    dr_dh = calculate_dr_dh(r, R_d)


        # Euler integration step
    r_new = r + dr_dh * dh
    h_new = h + dh

        # Store new values
    r_values.append(r_new)
    h_values.append(h_new)

    # Update current position
    r = r_new
    h = h_new

    return np.array(r_values), np.array(h_values)

def velocity(r_values):

    R_d = np.argmin(r_values)
    potential_energy = np.zeros_like(r_values)

    for i in range(1, R_d + 1):
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = PREM_I(segment_r)
        potential_energy[i] = np.trapz(segment_g, segment_r)

        # For ascending part: calculate separately as np.trapz needs increasing values
    for i in range(R_d + 1, len(r_values)):
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = PREM_I(segment_r)
        potential_energy[i] = np.trapz(segment_g, segment_r)


    # KE_current = PE_initial - PE_current (since KE_initial = 0)
    kinetic_energy = 0 - potential_energy
    # v = sqrt(2*KE/m) where m cancels out if we assume unit mass
    velocities = np.sqrt(2 * kinetic_energy)
    return velocities


def travel_time(r_values, h_values):
    times = [0]
    total_time = 0
    for i in range(1, len(r_values)):
        r1, r2 = r_values[i - 1], r_values[i]
        h1, h2 = h_values[i - 1], h_values[i]
        v = velocity(r2)
        # Calculate path length for this segment
        dr = r2 - r1
        dh = h2 - h1
        ds = np.sqrt(dr ** 2 + (r1 * dh) ** 2)  # Arc length in polar coordinates

        # Time to traverse this segment
        dt = ds / v
        total_time += dt
        times.append(total_time)

    return times

def coords_to_radians(init_lat,init_lon, final_lat, final_lon):
    theta_init = np.radians(90 - init_lat)
    phi_init = np.radians(init_lon)
    theta_end = np.radians(90 - final_lat)
    phi_end = np.radians(final_lon)

    return theta_init, phi_init, theta_end, phi_end


def main():
    # Set maximum depth as fraction of Earth radius
    R_d = 0.5 * R_earth  # Tunnel through half radius

    # Integration step size
    dh = 0.001  # Radians

    # Calculate path
    r_values, h_values = calculate_path(R_d, dh)


    # Calculate velocities
    velocities = velocity(r_values, PREM_I)

    # Calculate travel time
    times = travel_time(r_values, h_values, velocities)

    # Convert to cartesian coordinates for plotting
    x_values = r_values * np.cos(h_values)
    y_values = r_values * np.sin(h_values)

    # Plot path
    plt.figure(figsize=(10, 10))
    plt.plot(x_values, y_values, 'b-', linewidth=2)

    # Plot Earth circle
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(R_earth * np.cos(theta), R_earth * np.sin(theta), 'k-', linewidth=1)

    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Path through Earth (Max depth: {R_d / 1000:.0f} km)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.savefig('earth_path.png')

    print(f"Total travel time: {times[-1] / 60:.2f} minutes")

main()









