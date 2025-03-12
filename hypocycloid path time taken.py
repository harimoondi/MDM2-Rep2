import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
R_m = 6371 * 1000 # Earth's radius in m
g = 9.81  # Acceleration due to gravity in m/s^2
s_bristol_beijing_m = 8200 * 1000 # Approximate surface distance between Bristol and Beijing in m

# Compute b based on the hypocycloid property
b_bristol_beijing_m = s_bristol_beijing_m / (2 * np.pi)  # From the relation θ_AB = s / R

# Parametric equations for the hypocycloid path
t = np.linspace(0, 2 * np.pi, 1000)  # Parameter t
x_br_bb = (R_m - b_bristol_beijing_m) * np.cos(t) + b_bristol_beijing_m * np.cos(((R_m - b_bristol_beijing_m) / b_bristol_beijing_m) * t)
y_br_bb = (R_m - b_bristol_beijing_m) * np.sin(t) - b_bristol_beijing_m * np.sin(((R_m - b_bristol_beijing_m) / b_bristol_beijing_m) * t)

# Compute velocity and acceleration along the path
r_values = np.sqrt(x_br_bb**2 + y_br_bb**2)  # Compute radial distances
velocity = np.sqrt(g * (R_m**2 - r_values**2) / R_m)  # 

acceleration = - g * r_values / R_m  # Apply analytical acceleration formula

# Compute travel time using the theoretical formula
T_bristol_beijing = np.sqrt((s_bristol_beijing_m / R_m) * ((2 * np.pi * R_m - s_bristol_beijing_m) / g))
T_bristol_beijing_minutes = T_bristol_beijing / 60  # Convert to minutes

# Generate animation for the hypocycloid path
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

# Animation function
def update(frame):
    train.set_data([x_br_bb[frame]], [y_br_bb[frame]])  # Update train position

    return train, 

ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True, repeat=False)
plt.show()

# Print the estimated travel time
print(f"Estimated travel time for Bristol to Beijing for uniform gravity: {T_bristol_beijing_minutes:.2f} minutes")
