import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
R_km = 6371  # Earth's radius in km
R_m = R_km * 1000  # Convert to meters
g = 9.81  # Acceleration due to gravity in m/s^2
s_london_ny_km = 5570  # Approximate surface distance between London and New York in km
s_london_ny_m = s_london_ny_km * 1000  # Convert to meters

# Compute b based on the hypocycloid property
b_london_ny_m = s_london_ny_m / (2 * np.pi)  # From the relation θ_AB = s / R

# Parametric equations for the hypocycloid path
t = np.linspace(0, 2 * np.pi, 1000)  # Parameter t
x_ln_ny = (R_m - b_london_ny_m) * np.cos(t) + b_london_ny_m * np.cos(((R_m - b_london_ny_m) / b_london_ny_m) * t)
y_ln_ny = (R_m - b_london_ny_m) * np.sin(t) - b_london_ny_m * np.sin(((R_m - b_london_ny_m) / b_london_ny_m) * t)

# Compute velocity and acceleration along the path
dt = t[1] - t[0]  # Time step approximation
vx = np.gradient(x_ln_ny, dt)
vy = np.gradient(y_ln_ny, dt)
velocity = np.sqrt(vx**2 + vy**2)

ax = np.gradient(vx, dt)
ay = np.gradient(vy, dt)
acceleration = np.sqrt(ax**2 + ay**2)

# Compute travel time using the theoretical formula
T_london_ny = np.sqrt((s_london_ny_m / R_m) * ((2 * np.pi * R_m - s_london_ny_m) / g))
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
print(f"Estimated travel time for London to New York: {T_london_ny_minutes:.2f} minutes")
