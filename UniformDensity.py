import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
R_EARTH = 6371e3  # Earth's radius in meters
G = 6.674e-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24  # Mass of the Earth in kg
NUM_FRAMES = 300  # Number of frames for the animation

# Set the desired surface distance between points A and B (in kilometers)
surface_distance_km = 15_000  # Adjust this value (e.g., 5000, 10000, 15000 km)
surface_distance = surface_distance_km * 1e3  # Convert to meters

# Compute total transit time based on idealized conditions
T_TOTAL = np.pi * np.sqrt(R_EARTH ** 3 / (G * M))  # ~42 minutes (in seconds)

# Compute uniform density for visualization
uniform_density = M / ((4 / 3) * np.pi * R_EARTH ** 3)

def gravity_train_motion(surface_distance, num_frames=NUM_FRAMES):
    """Compute motion along a chord under idealized conditions (uniform density)."""
    theta = surface_distance / R_EARTH  # Central angle in radians
    chord_length = 2 * R_EARTH * np.sin(theta / 2)

    # Endpoints A and B on the Earth's surface
    A = np.array([-R_EARTH * np.sin(theta / 2), R_EARTH * np.cos(theta / 2)])
    B = np.array([R_EARTH * np.sin(theta / 2), R_EARTH * np.cos(theta / 2)])

    # Time values from 0 to T_TOTAL
    t_vals = np.linspace(0, T_TOTAL, num_frames)

    # Position arrays
    x_vals = np.zeros(num_frames)
    y_vals = np.zeros(num_frames)
    density_vals = np.full(num_frames, uniform_density)  # Constant density for visualization

    # Compute motion using simple harmonic equations
    for i in range(num_frames):
        t_norm = t_vals[i] / T_TOTAL * np.pi  # Normalize time to 0 to π
        s = chord_length / 2 * (1 - np.cos(t_norm))  # SHM displacement formula

        # Interpolate between A and B
        x_vals[i] = A[0] + (B[0] - A[0]) * (s / chord_length)
        y_vals[i] = A[1] + (B[1] - A[1]) * (s / chord_length)

    return t_vals, x_vals, y_vals, A, B, T_TOTAL, density_vals, chord_length

# Compute motion for the given surface distance
t_vals, x_vals, y_vals, A, B, T_TOTAL, density_vals, chord_length = gravity_train_motion(surface_distance)

# Set up figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Earth and train motion
ax1.set_xlim(-R_EARTH, R_EARTH)
ax1.set_ylim(-R_EARTH, R_EARTH)
ax1.set_aspect('equal')
ax1.axis('off')

# Earth outline
earth_circle = plt.Circle((0, 0), R_EARTH, color='black', fill=False)
ax1.add_patch(earth_circle)

# Chord (tunnel)
ax1.plot([A[0], B[0]], [A[1], B[1]], 'k-')

# Train (moving point)
train, = ax1.plot([], [], 'ro', markersize=6)

# Labels for endpoints
ax1.text(A[0] - 2e5, A[1] + 2e5, 'A', fontsize=12, color='blue')
ax1.text(B[0] + 2e5, B[1] - 2e5, 'B', fontsize=12, color='blue')

# Timestamp
timestamp = ax1.text(0, -R_EARTH * 0.95, '', fontsize=12, color='green')

# Plot 2: Density along the chord
ax2.set_xlim(0, chord_length)
ax2.set_ylim(0, uniform_density * 1.1)
ax2.set_xlabel('Distance along chord (m)', fontsize=14)
ax2.set_ylabel('Density (kg/m^3)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
density_line, = ax2.plot([], [], 'b-', label='Uniform Density', linewidth=2)
ax2.legend()

# Animation update function
def update_frame(i):
    train.set_data([x_vals[i]], [y_vals[i]])  # Update train position
    timestamp.set_text(f'Time: {t_vals[i]:.1f} s')  # Update timestamp
    density_line.set_data(np.linspace(0, chord_length, NUM_FRAMES)[:i], density_vals[:i])  # Update density plot
    return train, timestamp, density_line

# Create animation
ani = animation.FuncAnimation(fig, update_frame, frames=len(t_vals), interval=50, blit=True, repeat=False)

plt.tight_layout()
plt.show()
