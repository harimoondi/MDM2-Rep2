import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
R_EARTH = 6371e3  # Radius of Earth in meters
G = 6.674e-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24  # Mass of the Earth in kg
NUM_FRAMES = 300  # Number of frames in animation

# Function to calculate gravitational acceleration along the chord
def gravity_train_motion(surface_distance, num_frames=NUM_FRAMES):
    theta = surface_distance / R_EARTH  # Central angle
    chord_length = 2 * R_EARTH * np.sin(theta / 2)  # Chord length

    # Define tunnel endpoints
    A = np.array([-R_EARTH * np.sin(theta / 2), R_EARTH * np.cos(theta / 2)])
    B = np.array([R_EARTH * np.sin(theta / 2), R_EARTH * np.cos(theta / 2)])

    # Compute total travel time
    T_TOTAL = np.pi * np.sqrt(R_EARTH**3 / (G * M)) * (chord_length / (2 * R_EARTH))

    # Generate time values
    t_vals = np.linspace(0, T_TOTAL, num_frames)

    # Arrays for position and velocity
    x_vals = np.zeros(num_frames)
    y_vals = np.zeros(num_frames)
    velocity_vals = np.zeros(num_frames)

    # Simulating train motion
    for i in range(num_frames):
        t_norm = t_vals[i] / T_TOTAL * np.pi  # Normalize time (0 to π)
        s = chord_length / 2 * (1 - np.cos(t_norm))  # Sinusoidal motion
        v = (chord_length / 2) * (np.pi / T_TOTAL) * np.sin(t_norm)  # Velocity

        # Compute position along the chord
        x_vals[i] = A[0] + (B[0] - A[0]) * (s / chord_length)
        y_vals[i] = A[1] + (B[1] - A[1]) * (s / chord_length)
        velocity_vals[i] = v  # Store velocity

    return t_vals, x_vals, y_vals, velocity_vals, A, B, T_TOTAL, chord_length

# Define surface distance for the train
surface_distance = 8000e3  # meters

# Compute train motion data
t_vals, x_vals, y_vals, velocity_vals, A, B, T_TOTAL, chord_length = gravity_train_motion(surface_distance)

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Earth and train motion
ax1.set_xlim(-R_EARTH, R_EARTH)
ax1.set_ylim(-R_EARTH, R_EARTH)
ax1.set_aspect('equal')
ax1.axis('off')

# Draw Earth
earth_circle = plt.Circle((0, 0), R_EARTH, color='black', fill=False)
ax1.add_patch(earth_circle)

# Draw tunnel
ax1.plot([A[0], B[0]], [A[1], B[1]], 'k-')

# Train (animated point)
train, = ax1.plot([], [], 'ro', markersize=6)

# Labels for A and B
ax1.text(A[0] - 2e5, A[1] + 2e5, 'A', fontsize=12, color='blue')
ax1.text(B[0] + 2e5, B[1] - 2e5, 'B', fontsize=12, color='blue')

# Timestamp label
timestamp = ax1.text(0, -R_EARTH * 0.95, '', fontsize=12, color='green')

# Plot 2: Velocity along the chord
ax2.set_xlim(0, chord_length)
ax2.set_ylim(0, max(velocity_vals) * 1.1)
ax2.set_xlabel('Distance along chord (m)', fontsize=14)
ax2.set_ylabel('Velocity (m/s)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
velocity_line, = ax2.plot([], [], 'r-', label='Velocity', linewidth=2)

# Animation update function
def update_frame(i):
    train.set_data([x_vals[i]], [y_vals[i]])  # Update train position
    timestamp.set_text(f'Time: {t_vals[i]:.1f} s')  # Update time label
    velocity_line.set_data(np.linspace(0, chord_length, NUM_FRAMES)[:i], velocity_vals[:i])
    return train, timestamp, velocity_line

# Run animation
ani = animation.FuncAnimation(fig, update_frame, frames=len(t_vals), interval=50, blit=True, repeat=False)

plt.tight_layout()
plt.show()
