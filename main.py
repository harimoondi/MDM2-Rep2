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
    # Compute theta (central angle) from surface distance
    theta = surface_distance / R_EARTH

    # Compute chord length
    chord_length = 2 * R_EARTH * np.sin(theta / 2)

    # Define start and end points A and B
    A = np.array([-R_EARTH * np.sin(theta / 2), R_EARTH * np.cos(theta / 2)])
    B = np.array([R_EARTH * np.sin(theta / 2), R_EARTH * np.cos(theta / 2)])

    # Compute total travel time for the specific chord length (rough estimate)
    T_FULL_DIAMETER = np.pi * np.sqrt(R_EARTH ** 3 / (G * M))  # Time for full diameter
    T_TOTAL = T_FULL_DIAMETER * (chord_length / (2 * R_EARTH))  # Adjust for chord length

    # Generate time values for simulation
    t_vals = np.linspace(0, T_TOTAL, num_frames)

    # Initial conditions
    v = 0  # initial velocity
    s = 0  # initial position (at A)
    a = 0  # initial acceleration

    # Arrays to store position and velocity
    x_vals = np.zeros(num_frames)
    y_vals = np.zeros(num_frames)

    # Simulate the motion with gravity
    for i in range(1, num_frames):
        # Compute the radial distance from Earth's center (based on position along the chord)
        s_radial = s / chord_length  # Normalized position along chord

        # Gravitational acceleration depends on position along the chord
        a_gravity = (G * M) / (R_EARTH ** 2) * np.cos(np.pi * s_radial)  # Approximation

        # Update velocity using the gravitational force
        v += a_gravity * (t_vals[1] - t_vals[0])  # Velocity update (simplified)

        # Update position using velocity (simple model)
        s += v * (t_vals[1] - t_vals[0])  # Position update along the chord

        # Ensure the position does not go beyond the limits of the chord
        if s > chord_length:
            s = chord_length
        elif s < 0:
            s = 0

        # Calculate the corresponding (x, y) position on the chord
        x_vals[i] = A[0] + (B[0] - A[0]) * (s / chord_length)
        y_vals[i] = A[1] + (B[1] - A[1]) * (s / chord_length)

    return t_vals, x_vals, y_vals, A, B, T_TOTAL  # Include total travel time


# Function to update animation
def update_frame(i):
    train.set_data([x_vals[i]], [y_vals[i]])  # Update train position
    timestamp.set_text(f'Time: {t_vals[i]:.1f} s')  # Update time label
    return train, timestamp


# Function to display total time at the end
def on_complete(*args):
    total_time_label.set_text(f'Total Time: {T_TOTAL:.1f} s')  # Show total time at the end


# Define surface distance (adjustable)
surface_distance = 8000e3  # in meters (Great-circle distance)

# Compute motion data with gravity effects
t_vals, x_vals, y_vals, A, B, T_TOTAL = gravity_train_motion(surface_distance)

# Set up plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-R_EARTH, R_EARTH)
ax.set_ylim(-R_EARTH, R_EARTH)
ax.set_aspect('equal')
ax.axis('off')

# Draw Earth
earth_circle = plt.Circle((0, 0), R_EARTH, color='black', fill=False)
ax.add_patch(earth_circle)

# Draw chord (tunnel)
ax.plot([A[0], B[0]], [A[1], B[1]], 'k-')

# Train (animated point)
train, = ax.plot([], [], 'ro', markersize=6)

# Labels for A and B
ax.text(A[0] - 2e5, A[1] + 2e5, 'A', fontsize=12, color='blue')
ax.text(B[0] + 2e5, B[1] - 2e5, 'B', fontsize=12, color='blue')

# Timestamp label
timestamp = ax.text(0, -R_EARTH * 0.95, '', fontsize=12, color='green')

# Total Time label (displayed at the end)
total_time_label = ax.text(0, -R_EARTH * 1.05, '', fontsize=12, color='red', ha='center')

# Run animation once, ensuring it stops at the end
ani = animation.FuncAnimation(fig, update_frame, frames=len(t_vals), interval=50, blit=True, repeat=False)

# Add callback to show total time at the end
ani.event_source.add_callback(on_complete)

plt.show() 