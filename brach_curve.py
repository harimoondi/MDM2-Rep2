import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from mpl_toolkits.mplot3d import Axes3D

# Load the gravity function
file_path = "Gravity and Density Functions/functions/earth_gravity_fit.pkl"
with open(file_path, "rb") as f:
    gravity_function = pickle.load(f)

# Define initial and final points (latitudes and longitudes)
latitude = 51.4545  # Bristol Latitude
longitude = -2.5879  # Bristol Longitude
final_lat = 39.9042  # Sao Paulo Latitude
final_lon = 116.4074  # Sao Paulo Longitude

# Convert to radians
theta_init = np.radians(90 - latitude)
phi_init = np.radians(longitude)
theta_end = np.radians(90 - final_lat)
phi_end = np.radians(final_lon)

# Generate the arrays
theta_values = np.linspace(theta_init, theta_end, 100)
phi_values = np.linspace(phi_init, phi_end, 100)

# Debug: Plot initial guess for path
plt.plot(np.degrees(theta_values), np.degrees(phi_values), label="Initial Path")
plt.xlabel('Latitude (degrees)')
plt.ylabel('Longitude (degrees)')
plt.title('Initial Guess for Brachistochrone Path')
plt.legend()
plt.show()


# Define the velocity function
def velocity(r_values):
    g_values = gravity_function(r_values / 1000)
    if np.any(np.isnan(g_values)):
        raise ValueError("Gravity function returned NaN values!")
    g_integral = cumulative_trapezoid(g_values, r_values, initial=0)
    return np.sqrt(2 * g_integral)


# Define the arc length function
def arc_length(r, theta, phi, dr, dtheta, dphi):
    return np.sqrt(dr ** 2 + r ** 2 * dtheta ** 2 + r ** 2 * np.sin(theta) ** 2 * dphi ** 2)


# Define the time integral function
def time_integral(r_values, theta_values, phi_values):
    v_values = velocity(r_values)
    print("MAX V_VALUE", max(v_values))
    v_values[v_values == 0] = np.nan  # Avoid division by zero in case of zero velocity
    dr = np.diff(r_values)
    dtheta = np.diff(theta_values)
    dphi = np.diff(phi_values)
    ds_values = arc_length(r_values[:-1], theta_values[:-1], phi_values[:-1], dr, dtheta, dphi)

    # Debug: Print intermediate values
    print(f"ds_values: {ds_values}")
    print(f"v_values: {v_values}")

    time_values = ds_values / v_values[:-1]
    total_time = np.sum(time_values)
    return total_time


# Function to optimize the path
def optimize_path():
    # Initial guess for r_values (start at Earth's surface, descend to 6301 km, then ascend)
    r_initial = np.concatenate([np.linspace(6371, 6301, 25), np.linspace(6301, 6371, 25)])

    # Debug: Check the initial r_values
    print(f"Initial r_values: {r_initial}")

    def objective(r_values):
        # Convert to radians for theta and phi
        theta_values = np.linspace(theta_init, theta_end, len(r_values))
        phi_values = np.linspace(phi_init, phi_end, len(r_values))

        # Debug: Print the r_values and theta/phi ranges for checking
        print(f"r_values in objective: {r_values}")
        print(f"theta_values: {theta_values}")
        print(f"phi_values: {phi_values}")

        # Calculate total time
        total_time = time_integral(r_values, theta_values, phi_values)
        return total_time

    # Run the optimization with bounds on r (Earth radius and a bit below)
    bounds = [(6301, 6371)] * len(r_initial)  # r between 6301 km and 6371 km (just below and above Earth's surface)

    result = minimize(objective, r_initial, bounds=bounds, options={'disp': True, 'maxiter': 1000, 'tol': 1e-6})

    # Check optimization result
    if np.any(np.isnan(result.x)):
        raise ValueError("Optimization resulted in NaN values!")

    optimized_r = result.x
    return optimized_r


# Main execution
optimized_r = optimize_path()

# Debug: Print the final optimized radius values
print(f"Optimized r_values: {optimized_r}")

# Plotting (optional, you can plot the final path here if you want)
