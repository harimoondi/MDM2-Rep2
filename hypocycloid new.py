import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import quad  # Import the quad function for integration
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

# Define function for travel time integration
def integrand(r):
    g_r = gravity_function(r)
    if g_r <= 0:  # Prevent division by zero or negative sqrt
        return np.inf
    return 1 / np.sqrt(2 * g_r * (R_m - r))

# Integrate travel time numerically
T_london_ny, _ = quad(integrand, min(r_values), max(r_values))
T_london_ny_minutes = T_london_ny / 60  # Convert to minutes

# Create figure and axis for animation
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the Earth as a circle
earth_circle = plt.Circle((0, 0), R_m, color='lightblue', alpha=0.5, label="Earth")
ax.add_patch(earth_circle)

# Plot the hypocycloid path
ax.plot(x_ln_ny, y_ln_ny, label="Hypocycloid Path (London to NY)", color='orange')

# Mark start and end points
ax.scatter([x_ln_ny[0], x_ln_ny[-1]], [y_ln_ny[0], y_ln_ny[-1]], color='red', zorder=3, label="Start & End Points")

# Labels and title
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Gravity Train Hypocycloid Path: London to New York")

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
    train.set_data([x_ln_ny[frame]], [y_ln_ny[frame]])  # Update train position
    time_label.set_text(f'Time: {frame * (T_london_ny / len(t)):.1f} s')  # Update time label
    return train, time_label

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True, repeat=False)

# Show plot
plt.show()

# Print the estimated travel time
print(f"Estimated travel time for London to New York (using variable gravity): {T_london_ny_minutes:.2f} minutes")