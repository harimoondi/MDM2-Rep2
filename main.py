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

    # Compute total travel time for the specific chord length (exact solution)
    T_TOTAL = np.pi * np.sqrt(R_EARTH**3 / (G * M)) * (chord_length / (2 * R_EARTH))

    # Generate time values for simulation
    t_vals = np.linspace(0, T_TOTAL, num_frames)

    # Arrays to store position, velocity, density, and gravity
    x_vals = np.zeros(num_frames)
    y_vals = np.zeros(num_frames)
    density_vals = np.zeros(num_frames)
    gravity_vals = np.zeros(num_frames)

    # Density model: linear increase from surface to core
    def density(r):
        # r is the radial distance from Earth's center
        # Surface density: ~3000 kg/m^3, Core density: ~13000 kg/m^3
        return 3000 + (13000 - 3000) * (1 - r / R_EARTH)

    # Gravitational force at a given radial distance (Corrected)
    def gravity(r):
        if r == 0:
            return 0  # Gravity is zero at the center

        dr = 1000  # Integration step size (adjust as needed)
        mass_enclosed = 0

        for ri in np.arange(0, r, dr):  # Corrected: Exclude r itself
            mass_enclosed += density(ri) * 4 * np.pi * ri**2 * dr

        return G * mass_enclosed / r**2

    # Simulate the motion with gravity
    for i in range(num_frames):
        # Normalized time (0 to pi for one full oscillation)
        t_norm = t_vals[i] / T_TOTAL * np.pi

        # Position along the chord (sinusoidal motion)
        s = chord_length / 2 * (1 - np.cos(t_norm))

        # Calculate the corresponding (x, y) position on the chord
        x_vals[i] = A[0] + (B[0] - A[0]) * (s / chord_length)
        y_vals[i] = A[1] + (B[1] - A[1]) * (s / chord_length)

        # Compute the radial distance from Earth's center (Corrected)
        r = np.sqrt(x_vals[i]**2 + y_vals[i]**2)

        # Calculate density and gravity at this radial distance
        density_vals[i] = density(r)
        gravity_vals[i] = gravity(r)

    return t_vals, x_vals, y_vals, A, B, T_TOTAL, density_vals, gravity_vals, chord_length


# Function to update animation
def update_frame(i):
    train.set_data([x_vals[i]], [y_vals[i]])  # Update train position
    timestamp.set_text(f'Time: {t_vals[i]:.1f} s')  # Update time label
    density_line.set_data(np.linspace(0, chord_length, NUM_FRAMES)[:i], density_vals[:i])
    return train, timestamp, density_line  # Removed gravity_line from return


# Function to display total time at the end (Modified)
def on_complete(*args):
    # Total time is no longer displayed at the end.
    pass  # Do nothing at the end of the animation


# Define surface distance (adjustable)
surface_distance = 8000e3  # in meters (Great-circle distance)

# Compute motion data with gravity effects
t_vals, x_vals, y_vals, A, B, T_TOTAL, density_vals, _, chord_length = gravity_train_motion(surface_distance)  # _ to ignore gravity data

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))  # Only 2 subplots now

# Plot 1: Earth and Train Motion
ax1.set_xlim(-R_EARTH, R_EARTH)
ax1.set_ylim(-R_EARTH, R_EARTH)
ax1.set_aspect('equal')
ax1.axis('off')

# Draw Earth
earth_circle = plt.Circle((0, 0), R_EARTH, color='black', fill=False)
ax1.add_patch(earth_circle)

# Draw chord (tunnel)
ax1.plot([A[0], B[0]], [A[1], B[1]], 'k-')

# Train (animated point)
train, = ax1.plot([], [], 'ro', markersize=6)

# Labels for A and B
ax1.text(A[0] - 2e5, A[1] + 2e5, 'A', fontsize=12, color='blue')
ax1.text(B[0] + 2e5, B[1] - 2e5, 'B', fontsize=12, color='blue')

# Timestamp label
timestamp = ax1.text(0, -R_EARTH * 0.95, '', fontsize=12, color='green')

# Plot 2: Density along the chord
ax2.set_xlim(0, chord_length)
ax2.set_ylim(0, 14000)
ax2.set_xlabel('Distance along chord (m)', fontsize=14)
ax2.set_ylabel('Density (kg/m^3)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
density_line, = ax2.plot([], [], 'b-', label='Density', linewidth=2)

# Run animation once, ensuring it stops at the end
ani = animation.FuncAnimation(fig, update_frame, frames=len(t_vals), interval=50, blit=True, repeat=False)

# Add callback to show total time at the end
ani.event_source.add_callback(on_complete)

plt.tight_layout()
plt.show()