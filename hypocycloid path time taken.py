import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
R_km = 6371  # Earth's radius in km
R_m = R_km * 1000  # Convert to meters
g = 9.81  # Acceleration due to gravity in m/s^2
s_bristol_beijing_km = 8200  # Approximate surface distance between Bristol and Beijing in km
s_bristol_beijing_m = s_bristol_beijing_km * 1000  # Convert to meters

# Compute b based on the hypocycloid property
b_bristol_beijing_m = s_bristol_beijing_m / (2 * np.pi)  # From the relation θ_AB = s / R

# Parametric equations for the hypocycloid path
t = np.linspace(0, 2 * np.pi, 1000)  # Parameter t
x_br_bb = (R_m - b_bristol_beijing_m) * np.cos(t) + b_bristol_beijing_m * np.cos(((R_m - b_bristol_beijing_m) / b_bristol_beijing_m) * t)
y_br_bb = (R_m - b_bristol_beijing_m) * np.sin(t) - b_bristol_beijing_m * np.sin(((R_m - b_bristol_beijing_m) / b_bristol_beijing_m) * t)

# Compute velocity along the path
dt = t[1] - t[0]  # Time step approximation
vx = np.gradient(x_br_bb, dt)
vy = np.gradient(y_br_bb, dt)
velocity = np.sqrt(vx**2 + vy**2)

# Compute travel time using the theoretical formula
T_bristol_beijing = np.sqrt((s_bristol_beijing_m / R_m) * ((2 * np.pi * R_m - s_bristol_beijing_m) / g))
T_bristol_beijing_minutes = T_bristol_beijing / 60  # Convert to minutes

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
train, = ax1.plot([], [], 'ro', markersize=6)  # Train marker

# Velocity-Time Graph on second subplot
ax2.set_title("Velocity vs Time")
ax2.set_xlabel("Time Index")
ax2.set_ylabel("Velocity (m/s)")
ax2.set_xlim(0, len(t))
ax2.set_ylim(0, np.max(velocity) * 1.1)
velocity_line, = ax2.plot([], [], 'b-', label="Velocity")
ax2.legend()
ax2.grid(True)

# Animation function
def update(frame):
    train.set_data([x_br_bb[frame]], [y_br_bb[frame]])  # Update train position
    velocity_line.set_data(range(frame + 1), velocity[:frame + 1])  # Update velocity plot
    return train, velocity_line

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True, repeat=False)

# Show plot
plt.tight_layout()
plt.show()

# Print the estimated travel time
print(f"Estimated travel time for Bristol to Beijing: {T_bristol_beijing_minutes:.2f} minutes")
