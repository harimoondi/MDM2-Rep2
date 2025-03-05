
'''
This script is used to attempt to validate the model. We know that the brachistochrone curve
is the fastest path through a uniform gravity field. So we set out a unifrom gravitational field ( 9.81 everywhere),
and plot the brachichrone curve, then make an initial guess near this curve and the minimizer should return something
very close to the theoretical brachistochrone curve. MSE is used to calculate the error between the two curves. (should be small)
'''






import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
# Constants
r_earth = 6371000  # Earth radius in meters
G = const.G
M = 5.972e24
R = 6371000


def central_angle_calculation(theta_init, phi_init, theta_end, phi_end):
    return np.arccos(np.sin(theta_init) * np.sin(theta_end) +
                     np.cos(theta_init) * np.cos(theta_end) * np.cos(phi_end - phi_init))

def uniform_gravity_function(r):
    return np.where(r == 0, 0, G * M / R ** 2)

def calculate_ep(r: list): # Used to validate np.trapz is integrating correctly
    r = np.array(r)
    return np.where(r == 0, 0, G * M * (r - R) / R ** 2)

def velocity(r_values):
    min_depth_index = np.argmin(r_values)
    potential_energy = np.zeros_like(r_values)

    for i in range(1, min_depth_index + 1):
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = uniform_gravity_function(segment_r)
        potential_energy[i] = np.trapz(segment_g, segment_r)

    for i in range(min_depth_index + 1, len(r_values)):
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = uniform_gravity_function(segment_r)
        potential_energy[i] = np.trapz(segment_g, segment_r)


    kinetic_energy = 0 - potential_energy
    kinetic_energy = np.maximum(kinetic_energy, 0)
    velocities = np.sqrt(2 * kinetic_energy)

    return velocities

def arc_length(r, dr, dalpha):
    return np.sqrt(dr ** 2 + r ** 2 * dalpha ** 2)

def time_integral(r_values, alpha_values):
    v_values = velocity(r_values)
    dr_values = np.diff(r_values)
    dalpha_values = np.diff(alpha_values)
    time_values = np.zeros(len(dr_values))

    for i in range(len(dr_values)):
        if v_values[i] == 0:
            time_values[i] = 0
        else:
            ds = arc_length(r_values[i], dr_values[i], dalpha_values[i])
            time_values[i] = ds / v_values[i]

    total_time = np.sum(time_values)
    return total_time

def objective(params, alpha_values):
    r_values = params
    r_values[0] = r_values[-1] = r_earth
    total_time = time_integral(r_values, alpha_values)
    return total_time

def optimize_path(r_init, alpha_values):
    n_points = len(r_init)
    bounds = [(0.1 * r_earth, r_earth) for _ in range(n_points)]
    bounds[0] = bounds[-1] = (r_earth, r_earth)

    result = minimize(
        objective,
        r_init,
        args=(alpha_values,),
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'disp': True,
            'maxiter': 1000,
            'ftol': 1e-8,
            'gtol': 1e-8,
            'maxcor': 50
        }
    )

    optimized_r_values = result.x
    optimized_r_values[0] = optimized_r_values[-1] = r_earth
    return optimized_r_values

def theoretical_brachistochrone(R):
    t_values = np.linspace(0, np.pi, 200)
    x = R * (t_values - np.sin(t_values))
    y = R * (1 - np.cos(t_values))


    magnitude = np.array(np.sqrt(x ** 2 + y ** 2))
    # Scale Magnitude
    scale_mag = R / magnitude[-1]
    magnitude = magnitude * scale_mag

    x_scale = R / x[-1]
    y_scale = R / y[-1]
    x = x * x_scale
    y = y * y_scale
    y = -y
    y = y + R
    return x, y, magnitude


def close_initial_guess(R):
    # Define fixed points
    x, y, magnitude = theoretical_brachistochrone(R)
    fixed_indices = [0, 25, 50, 100, 150, 175, 199]
    # fixed_values = [R, magnitude[25], magnitude[50], magnitude[100], magnitude[150], magnitude[175], R]
    fixed_values = [R,  4500000, 3400000, 3500000, 3400000, 4500000,  R]

    # Create cubic spline interpolation
    spline = CubicSpline(fixed_indices, fixed_values, bc_type='clamped')

    # Generate smooth curve
    r_initial = spline(np.arange(200))

    # Add small noise for variability
    r_initial += np.random.uniform(-1e-2, 1e-2, size=r_initial.shape)

    return r_initial


# Coordinates
init_lat = 90.0  # North Pole
init_lon = 0.0  # Prime Meridian
final_lat = 0.0  # Equator
final_lon = 90.0  # 90 degrees East

# Convert to radians
theta_init = np.radians(90 - init_lat)
phi_init = np.radians(init_lon)
theta_end = np.radians(90 - final_lat)
phi_end = np.radians(final_lon)

# Initial guess
r_initial = close_initial_guess(R)
print('R_initial' , r_initial)

# Generate alpha values
central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
alpha_values = np.linspace(0, central_angle, len(r_initial))

# Plotting
plt.figure(figsize=(12, 12))


# Optimize the path
optimized_r = optimize_path(r_initial, alpha_values)
travel_time = time_integral(optimized_r, alpha_values)
max_depth = r_earth - np.min(optimized_r)

# Plot the optimized tunnel path
plt.plot(optimized_r * np.cos(alpha_values), optimized_r * np.sin(alpha_values), 'r-', linewidth=2, label='Optimized Tunnel')

# Calculate the Numerical Results
# Calculate MSE between optimized curve and theoretical brachistochrone
x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
theoretical_r = np.sqrt(x_theoretical ** 2 + y_theoretical ** 2)
interp_func = interp1d(np.linspace(0, 1, len(theoretical_r)), theoretical_r)
resampled_theoretical_r = interp_func(np.linspace(0, 1, len(optimized_r)))
mse = np.mean((optimized_r - resampled_theoretical_r) ** 2)
print(f'Mean Squared Error (MSE) between optimized and theoretical curve: {mse:.4f}')


# Print the results
print("Travel Time (mins):", travel_time / 60)
print("Max Depth (m):", max_depth)


# Plot the earths surface
theta = np.linspace(0, 2 * np.pi, 1000)
x_circle = r_earth * np.cos(theta)
y_circle = r_earth * np.sin(theta)
plt.plot(x_circle, y_circle, 'k-', linewidth=1)

# Plot the theoretical brachistochrone
x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
plt.plot(x_theoretical, y_theoretical, 'm--', linewidth=2, label='Theoretical Brachistochrone')

# Plot initial guess
plt.plot(r_initial * np.cos(alpha_values), r_initial * np.sin(alpha_values), 'b--', linewidth=1, label='Initial Guess')

# Mark the start and end points
plt.plot(0, r_earth, 'ro', label='Start')
plt.plot(r_earth, 0, 'go', label='End')

plt.title('Model Validation by Comparing Optimized Tunnel in a '
          'Uniform Gravity Field to Theoretical Brachistochrone')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

