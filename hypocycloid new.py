import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from scipy.integrate import cumulative_trapezoid

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

# Load pre-computed gravity model (fit to PREM model)
earth_gravity_fit = '/Users/gainsong/Desktop/MDM2-Rep2/Gravity and Density Functions/functions/earth_gravity_fit.pkl'
with open(earth_gravity_fit, 'rb') as f:
    gravity_function = pickle.load(f)

# Constants
R_m = 6371 * 1000  # Earth's radius in m
s_bristol_beijing_m = 8200 * 1000 # Approximate surface distance between Bristol and Beijing in m
g_surface = gravity_function(R_m)  # Gravity at Earth's surface

# Compute travel time equation 
T_bristol_beijing = np.sqrt((s_bristol_beijing_m / R_m) * ((2 * np.pi * R_m - s_bristol_beijing_m) / g_surface))

# Convert to minutes
T_bristol_beijing_minutes = T_bristol_beijing / 60

# Print the estimated travel time
print(f"Estimated travel time for Bristol to Beijing: {T_bristol_beijing_minutes:.2f} minutes")

# Compute b based on the hypocycloid property
b_bristol_beijing_m = s_bristol_beijing_m / (2 * np.pi)

# Parametric equations for the hypocycloid path
t = np.linspace(0, 2 * np.pi, 1000)  # Parameter t
x_br_bb = (R_m - b_bristol_beijing_m) * np.cos(t) + b_bristol_beijing_m * np.cos(((R_m - b_bristol_beijing_m) / b_bristol_beijing_m) * t)
y_br_bb = (R_m - b_bristol_beijing_m) * np.sin(t) - b_bristol_beijing_m * np.sin(((R_m - b_bristol_beijing_m) / b_bristol_beijing_m) * t)

# Compute radius at each point
r_values = np.sqrt(x_br_bb**2 + y_br_bb**2)

# Get corresponding gravity values
g_values = np.array([gravity_function(r) for r in r_values])

# Ensure no division by zero in travel time calculation
g_values[g_values == 0] = 1e-6  # Replace zero gravity with a small positive value

# Generate the hypocycloid path animation
fig, ax = plt.subplots(figsize=(6, 6))

earth_circle = plt.Circle((0, 0), R_m, color='lightblue', alpha=0.5, label="Earth")
ax.add_patch(earth_circle)

# Plot the hypocycloid path
ax.plot(x_br_bb, y_br_bb, label="Hypocycloid Path (Bristol to Beijing)", color='orange')
ax.scatter([x_br_bb[0], x_br_bb[-1]], [y_br_bb[0], y_br_bb[-1]], color='red', zorder=3, label="Start & End Points")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Gravity Train Hypocycloid Path: Bristol to Beijing")

# Set axis limits to match Earth's scale
ax.set_xlim(-R_m * 1.1, R_m * 1.1)
ax.set_ylim(-R_m * 1.1, R_m * 1.1)
ax.set_aspect('equal')
ax.legend()
ax.grid(True)

# Animation setup
train, = ax.plot([], [], 'ro', markersize=6)  # Train marker
time_label = ax.text(-R_m * 0.9, -R_m * 0.9, '', fontsize=12, color='black')

# Animation function
def update(frame):
    train.set_data([x_br_bb[frame]], [y_br_bb[frame]])  # Update train position
    time_label.set_text(f'Time: {frame * (T_bristol_beijing / len(t)):.1f} s')  # Update time label
    return train, time_label

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True, repeat=False)
plt.show()

# Compute velocity using energy conservation
potential_energy = cumulative_trapezoid(g_values, r_values, initial=0)
velocity = np.sqrt(2 * np.abs(potential_energy))  # Ensure non-negative values

# Prevent division by zero in acceleration calculation
epsilon = 1e-6
denominator = (R_m - r_values) + epsilon  # Small epsilon added to prevent division by zero

# Compute acceleration safely
acceleration = g_values - (velocity**2 / denominator)

# Remove NaN values before finding max
valid_acceleration = acceleration[~np.isnan(acceleration)]
valid_velocity = velocity[~np.isnan(velocity)]

# Find maximum velocity and acceleration
max_velocity = np.max(valid_velocity)
max_acceleration = np.max(valid_acceleration)

# Print the results
print(f"Maximum velocity: {max_velocity:.2f} m/s")
print(f"Maximum acceleration: {max_acceleration:.2f} m/s²")







'''velocity graph'''

g = 9.81  # Constant g for the uniform model

# Parametric equations for the hypocycloid path
t = np.linspace(0, 2 * np.pi, 1000)  # Parameter t

# Get corresponding gravity values from PREM model
g_values_prem = np.array([gravity_function(r) for r in r_values])

# Compute velocity using energy conservation with PREM gravity
potential_energy_prem = cumulative_trapezoid(g_values_prem, r_values, initial=0)
velocity_prem = np.sqrt(2 * np.abs(potential_energy_prem))  # Ensure non-negative values

# Compute velocity for the constant g model
velocity_const_g = np.sqrt(g * (R_m**2 - r_values**2) / R_m)

# Compute depth (distance below Earth's surface)
depth_values = R_m - r_values

# Generate time axis for comparison
T_bristol_beijing = np.sqrt((s_bristol_beijing_m / R_m) * ((2 * np.pi * R_m - s_bristol_beijing_m) / g_surface))
time_values_prem = np.linspace(0, T_bristol_beijing, len(velocity_prem))  # PREM time axis
time_values_const_g = np.linspace(0, T_bristol_beijing, len(velocity_const_g))  # Constant g time axis

# Plot the grahs
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot velocity vs. time for both models
axs[0].plot(time_values_const_g / 60, velocity_const_g, label="Constant g Model", color='b', linestyle='--')  
axs[0].plot(time_values_prem / 60, velocity_prem, label="PREM Gravity Model", color='r')  
axs[0].set_xlabel("Time (minutes)", fontsize=15)
axs[0].set_ylabel("Velocity (m/s)", fontsize=15)
axs[0].set_title("Velocity vs. Time", fontsize=15)
axs[0].legend()
axs[0].grid(True)

# Plot velocity vs. depth for both models
axs[1].plot(depth_values / 1000, velocity_prem, label="PREM Gravity Model", color='r')  # Convert depth to km
axs[1].plot(depth_values / 1000, velocity_const_g, label="Constant g Model", color='b', linestyle='--')  # Convert depth to km
axs[1].set_xlabel("Depth (km)",fontsize=15)
axs[1].set_ylabel("Velocity (m/s)",fontsize=15)
axs[1].set_title("Velocity vs. Depth",fontsize=15)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


