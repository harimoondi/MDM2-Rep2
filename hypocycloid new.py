import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import cumulative_trapezoid
import pickle

# Define DensityFunction class needed for unpickling
class DensityFunction:
    def __init__(self, input_function, surface_radius=6371000):
        self.input_function = input_function
        self.surface_radius = surface_radius
    
    def __call__(self, r):
        if np.isscalar(r):
            return self.input_function(r) if r <= self.surface_radius else 0
        else:
            return np.where(r <= self.surface_radius, self.input_function(r), 0)

# Load pre-computed gravity model
earth_gravity_fit = '/Users/gainsong/Desktop/MDM2-Rep2/Gravity and Density Functions/functions/earth_gravity_fit.pkl'
with open(earth_gravity_fit, 'rb') as f:
    gravity_function = pickle.load(f)

# Constants
R_km = 6371  # Earth's radius in km
R_m = R_km * 1000  # Convert to meters
s_london_ny_km = 5570  # Approximate surface distance between London and New York in km
s_london_ny_m = s_london_ny_km * 1000  # Convert to meters
g_surface = gravity_function(R_m)  # Gravity at Earth's surface

# Compute travel time using the equation from the image
T_london_ny = np.sqrt((s_london_ny_m / R_m) * ((2 * np.pi * R_m - s_london_ny_m) / g_surface))

# Convert to minutes
T_london_ny_minutes = T_london_ny / 60

# Print the estimated travel time
print(f"Estimated travel time for London to New York (using analytical formula): {T_london_ny_minutes:.2f} minutes")

# Compute b based on the hypocycloid property
b_london_ny_m = s_london_ny_m / (2 * np.pi)

# Parametric equations for the hypocycloid path
t = np.linspace(0, 2 * np.pi, 1000)  # Parameter t
x_ln_ny = (R_m - b_london_ny_m) * np.cos(t) + b_london_ny_m * np.cos(((R_m - b_london_ny_m) / b_london_ny_m) * t)
y_ln_ny = (R_m - b_london_ny_m) * np.sin(t) - b_london_ny_m * np.sin(((R_m - b_london_ny_m) / b_london_ny_m) * t)

# Compute radius at each point
r_values = np.sqrt(x_ln_ny**2 + y_ln_ny**2)

# Get corresponding gravity values
g_values = np.array([gravity_function(r) for r in r_values])

# Ensure no division by zero in travel time calculation
g_values[g_values == 0] = 1e-6  # Replace zero gravity with a small positive value

# Compute velocity using energy conservation
potential_energy = cumulative_trapezoid(g_values, r_values, initial=0)
velocity = np.sqrt(2 * np.abs(potential_energy))  # Ensure non-negative values

# Compute acceleration
acceleration = g_values - (velocity**2 / (R_m - r_values))

# Find maximum velocity and maximum acceleration
max_velocity = np.max(velocity)
max_acceleration = np.max(acceleration)

# Print the results
print(f"Maximum velocity: {max_velocity:.2f} m/s")
print(f"Maximum acceleration: {max_acceleration:.2f} m/s²")
