import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
import plotly.graph_objects as go

# Load the gravity function
file_path = "Gravity and Density Functions/functions/earth_gravity_fit.pkl"
with open(file_path, "rb") as f:
    gravity_function = pickle.load(f)

r_earth = 6371000  # Earth radius in meters

# Define initial and final points with correct coordinates
# Bristol: 51.4545° N, 2.5879° W
# Sao Paulo: 23.5558° S, 46.6396° W
init_lat = 51.4545   # North is positive
init_lon = -2.5879   # West is negative
final_lat = -23.5558  # South is negative
final_lon = -46.6396  # West is negative

# Convert to radians
theta_init = np.radians(90 - init_lat)
phi_init = np.radians(init_lon)
theta_end = np.radians(90 - final_lat)
phi_end = np.radians(final_lon)

# Generate the arrays - start with an initial guess for path
r_init = np.concatenate([np.linspace(r_earth, r_earth - 70000, 25), np.linspace(r_earth - 70000, r_earth, 25)])
theta_values = np.linspace(theta_init, theta_end, len(r_init))
phi_values = np.linspace(phi_init, phi_end, len(r_init))


# Define the velocity function using gravitational potential energy
# Define the velocity function using gravitational potential energy
def velocity(r_values):
    # First, find the minimum radius point (deepest point in tunnel)
    min_depth_index = np.argmin(r_values)
    # Calculate gravitational potential at each point
    # We're calculating potential energy difference relative to the surface
    # Gravity is acceleration, so we need to integrate g(r) * dr to get potential
    potential_energy = np.zeros_like(r_values)

    # For descending part: integrate from surface to current depth
    for i in range(1, min_depth_index + 1):
        # Small segments for integration
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = gravity_function(segment_r)
        # Potential energy is integral of g(r) dr from surface to current point
        potential_energy[i] = np.trapz(segment_g, segment_r)

    # For ascending part: calculate separately
    for i in range(min_depth_index + 1, len(r_values)):
        # Small segments for integration
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = gravity_function(segment_r)
        # Potential energy is integral of g(r) dr from surface to current point
        potential_energy[i] = np.trapz(segment_g, segment_r)


    # The kinetic energy at each point is the negative of potential energy
    # plus the initial kinetic energy (which is 0 at the start)
    # Conservation of energy: PE_initial + KE_initial = PE_current + KE_current
    # KE_current = PE_initial - PE_current (since KE_initial = 0)
    # We set datum PE = 0 at the surface
    kinetic_energy = 0 - potential_energy


    # Ensure non-negative kinetic energy (due to numerical errors)
    kinetic_energy = np.maximum(kinetic_energy, 0)

    # v = sqrt(2*KE/m) where m cancels out if we assume unit mass
    velocities = np.sqrt(2 * kinetic_energy)
    return velocities


# Define the arc length function
def arc_length(r, theta, phi, dr, dtheta, dphi):
    return np.sqrt(dr ** 2 + r ** 2 * dtheta ** 2 + r ** 2 * np.sin(theta) ** 2 * dphi ** 2)


# Define the time integral function
def time_integral(r_values, theta_values, phi_values):
    v_values = velocity(r_values)

    # Initialize the travel times array
    time_values = np.zeros(len(v_values) - 1)  # Time values for each segment

    # Handle the first time step explicitly: Set travel time to 0 when v_values[0] is 0
    if v_values[0] == 0:
        time_values[0] = 0
    else:
        # Calculate arc length for the first step
        dr = np.diff(r_values)[0]
        dtheta = np.diff(theta_values)[0]
        dphi = np.diff(phi_values)[0]
        ds = arc_length([r_values[0]], [theta_values[0]], [phi_values[0]], [dr], [dtheta], [dphi])
        time_values[0] = ds / v_values[0]  # Time for the first segment

    # For the rest of the steps, calculate time normally
    for i in range(1, len(v_values) - 1):
        if v_values[i] == 0:
            time_values[i] = 0  # Set travel time to 0 if velocity is zero
        else:
            dr = np.diff(r_values)[i]
            dtheta = np.diff(theta_values)[i]
            dphi = np.diff(phi_values)[i]
            ds = arc_length(r_values[i], theta_values[i], phi_values[i], dr, dtheta, dphi)
            time_values[i] = ds / v_values[i]  # Time for each segment

    # Calculate the total time
    total_time = np.sum(time_values)

    return total_time


# Function to optimize the path
def objective(params):
    # Reshape the parameters into the r_values array
    r_values = params
    r_values[0] = r_values[-1] = r_earth

    total_time = time_integral(r_values, theta_values, phi_values)

    return total_time


# Run the optimization with bounds on the entire path (multiple points)
def optimize_path():
    n_points = 50


    max_depth = 2000000  # 2000 km initial depth guess

    # # Create initial guess with a half sinusoidal profile
    # x = np.linspace(0, np.pi, n_points)
    # depth_profile = max_depth * np.sin(x)
    # initial_r_values = r_earth - depth_profile

    # Create initial guess with a linear profile
    initial_r_values = r_init = np.concatenate([np.linspace(r_earth, r_earth - 70000, 25), np.linspace(r_earth - 70000, r_earth, 25)])
    initial_params = initial_r_values

    # Define bounds to keep radius within reasonable limits
    # Allow deeper tunnels to explore more solutions
    min_radius = r_earth - 30000000  # Allow depths up to 30000 km
    bounds = [(min_radius, r_earth) for _ in range(n_points)]

    # Fix the endpoints to be at Earth's surface
    bounds[0] = (r_earth, r_earth)
    bounds[-1] = (r_earth, r_earth)





        # Use L-BFGS-B for bounded optimization
    result = minimize(
            objective,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'disp': True,
                'maxiter': 1000,  #  iterations
                'ftol': 1e-10,  #  function tolerance
                'gtol': 1e-8,  #  gradient tolerance
                'maxcor': 50  # Increase memory for approximation
            }
        )



    optimized_r_values = result.x

    # Ensure endpoints are exactly at Earth's surface
    optimized_r_values[0] = optimized_r_values[-1] = r_earth

    return optimized_r_values





# Main execution

optimized_r = optimize_path()

# Update theta and phi for the final path
theta_values = np.linspace(theta_init, theta_end, len(optimized_r))
phi_values = np.linspace(phi_init, phi_end, len(optimized_r))

# Calculate travel time for the optimized path
travel_time = time_integral(optimized_r, theta_values, phi_values)
print(f"Optimized travel time: {travel_time:.2f} seconds ({travel_time / 60:.2f} minutes)")

# Calculate tunnel depth
max_depth = r_earth - np.min(optimized_r)
print(f"Maximum tunnel depth: {max_depth:.2f} meters ({max_depth / 1000:.2f} km)")


# Convert spherical to cartesian coordinates for the tunnel path

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


# Calculate tunnel path
x_tunnel, y_tunnel, z_tunnel = spherical_to_cartesian(optimized_r, theta_values, phi_values)

# Calculate start and end points
x_start, y_start, z_start = spherical_to_cartesian(r_earth, theta_init, phi_init)
x_end, y_end, z_end = spherical_to_cartesian(r_earth, theta_end, phi_end)

# Define the Earth sphere using the approach provided
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_earth = r_earth * np.outer(np.cos(u), np.sin(v))
y_earth = r_earth * np.outer(np.sin(u), np.sin(v))
z_earth = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))

# Create the figure with the Earth and tunnel
fig = go.Figure()

# Add Earth surface
fig.add_trace(go.Surface(
    z=z_earth, x=x_earth, y=y_earth,
    opacity=0.3,
    surfacecolor=np.full_like(x_earth, 1),
    colorscale=[[0, 'blue'], [1, 'blue']],
    showscale=False,
    name='Earth'
))

# Add tunnel path
fig.add_trace(go.Scatter3d(
    x=x_tunnel, y=y_tunnel, z=z_tunnel,
    mode='lines',
    line=dict(color='red', width=6),
    name='Optimal Tunnel'
))

# Add start point (Bristol)
fig.add_trace(go.Scatter3d(
    x=[x_start], y=[y_start], z=[z_start],
    mode='markers',
    marker=dict(size=8, color='green'),
    name='Start (Bristol)'
))

# Add end point (Sao Paulo)
fig.add_trace(go.Scatter3d(
    x=[x_end], y=[y_end], z=[z_end],
    mode='markers',
    marker=dict(size=8, color='orange'),
    name='End (Sao Paulo)'
))

# Update the layout for better visualization
fig.update_layout(
    title=f"Optimal Gravity Tunnel Path<br>Travel time: {travel_time / 60:.1f} minutes | Depth: {max_depth / 1000:.1f} km",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data'
    ),
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.5)'
    ),
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=50)
)

# Show the figure
fig.show()
