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

# Ensure no division by zero in travel time calculation
g_values[g_values == 0] = 1e-6  # Replace zero gravity with a small positive value

# Create figure and axis for animation
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the Earth as a circle
earth_circle = plt.Circle((0, 0), R_m, color='lightblue', alpha=0.5, label="Earth")
ax.add_patch(earth_circle)

# Plot the hypocycloid path
ax.plot(x_br_bb, y_br_bb, label="Hypocycloid Path (Bristol to Beijing)", color='orange')

# Mark start and end points
ax.scatter([x_br_bb[0], x_br_bb[-1]], [y_br_bb[0], y_br_bb[-1]], color='red', zorder=3, label="Start & End Points")

# Labels and title
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Gravity Train Hypocycloid Path: Bristol to Beijing")

# Set axis limits to match Earth's scale
ax.set_xlim(-R_m * 1.1, R_m * 1.1)
ax.set_ylim(-R_m * 1.1, R_m * 1.1)
ax.set_aspect('equal')

# Add legend and grid
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

# Show plot
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
