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

# Load pre-computed gravity model
earth_gravity_fit = '/Users/gainsong/Desktop/MDM2-Rep2/Gravity and Density Functions/functions/earth_gravity_fit.pkl'
with open(earth_gravity_fit, 'rb') as f:
    gravity_function = pickle.load(f)

# Constants
R_km = 6371  # Earth's radius in km
R_m = R_km * 1000  # Convert to meters
s_bristol_beijing_km = 8200  # Approximate surface distance between Bristol and Beijing in km
s_bristol_beijing_m = s_bristol_beijing_km * 1000  # Convert to meters
g_surface = gravity_function(R_m)  # Gravity at Earth's surface

# Compute travel time using the equation from the image
T_bristol_beijing = np.sqrt((s_bristol_beijing_m / R_m) * ((2 * np.pi * R_m - s_bristol_beijing_m) / g_surface))

# Convert to minutes
T_bristol_beijing_minutes = T_bristol_beijing / 60

# Print the estimated travel time
print(f"Estimated travel time for Bristol to Beijing (using analytical formula): {T_bristol_beijing_minutes:.2f} minutes")

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

g_values[g_values == 0] = 1e-6  # Replace zero gravity with a small positive value

# Compute velocity using energy conservation
potential_energy = cumulative_trapezoid(g_values, r_values, initial=0)
velocity = np.sqrt(2 * np.abs(potential_energy))  # Ensure non-negative values

# Compute acceleration safely
epsilon = 1e-6
denominator = (R_m - r_values) + epsilon  # Small epsilon added to prevent division by zero
acceleration = g_values - (velocity**2 / denominator)

# Remove NaN values
valid_acceleration = acceleration[~np.isnan(acceleration)]
valid_velocity = velocity[~np.isnan(velocity)]

# Create figure and axes for animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))

# Plot the Earth as a circle on the first subplot
ax1.set_title("Gravity Train Hypocycloid Path: Bristol to Beijing")
earth_circle = plt.Circle((0, 0), R_m, color='lightblue', alpha=0.5, label="Earth")
ax1.add_patch(earth_circle)
ax1.plot(x_br_bb, y_br_bb, label="Hypocycloid Path", color='orange')
ax1.scatter([x_br_bb[0], x_br_bb[-1]], [y_br_bb[0], y_br_bb[-1]], color='red', zorder=3, label="Start & End Points")
ax1.set_xlim(-R_m * 1.1, R_m * 1.1)
ax1.set_ylim(-R_m * 1.1, R_m * 1.1)
ax1.set_aspect('equal')
ax1.legend()
ax1.grid(True)
train, = ax1.plot([], [], 'ro', markersize=6)

# Velocity-Time Graph on second subplot
ax2.set_title("Velocity vs Time")
ax2.set_xlabel("Time Index")
ax2.set_ylabel("Velocity (m/s)")
ax2.set_xlim(0, len(t))
ax2.set_ylim(0, np.max(valid_velocity) * 1.1)
velocity_line, = ax2.plot([], [], 'b-', label="Velocity")
ax2.legend()
ax2.grid(True)

# Animation function
def update(frame):
    train.set_data([x_br_bb[frame]], [y_br_bb[frame]])
    velocity_line.set_data(range(frame + 1), valid_velocity[:frame + 1])
    return train, velocity_line

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True, repeat=False)

# Show plot
plt.tight_layout()
plt.show()

# Print the results
print(f"Maximum velocity: {np.max(valid_velocity):.2f} m/s")
print(f"Maximum acceleration: {np.max(valid_acceleration):.2f} m/s²")
