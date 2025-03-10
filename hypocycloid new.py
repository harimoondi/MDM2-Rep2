import numpy as np
import scipy.integrate as spi
import geopy.distance
import matplotlib.pyplot as plt

# Constants
g = 9.81  # m/s² (gravity at Earth's surface)
R = 6371000  # m (Earth's radius)
b = R / 3  # Hypocycloid rolling circle radius

# Function to convert lat/lon to Cartesian coordinates (Earth-centered)
def latlon_to_cartesian(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])

# Function to compute arc angle between two cities
def compute_arc_angle(cityA, cityB):
    A = latlon_to_cartesian(*cityA)
    B = latlon_to_cartesian(*cityB)
    angle = np.arccos(np.dot(A, B) / (R**2))  # Arc angle in radians
    return angle

# Hypocycloid parametric equations
def x(t, theta):
    return (R - b) * np.cos(t) + b * np.cos(((R - b) / b) * t)

def y(t, theta):
    return (R - b) * np.sin(t) - b * np.sin(((R - b) / b) * t)

# First derivatives
def dx_dt(t, theta):
    return -(R - b) * np.sin(t) - b * ((R - b) / b) * np.sin(((R - b) / b) * t)

def dy_dt(t, theta):
    return (R - b) * np.cos(t) - b * ((R - b) / b) * np.cos(((R - b) / b) * t)

# Arc length element ds
def ds_dt(t, theta):
    return np.sqrt(dx_dt(t, theta)**2 + dy_dt(t, theta)**2)

# Velocity function using energy conservation
def velocity(r):
    return np.sqrt((g / R) * (R**2 - r**2))

# Travel time integral function
def travel_time_integrand(t, theta):
    r = np.sqrt(x(t, theta)**2 + y(t, theta)**2)  # Compute r from parametric equations
    return ds_dt(t, theta) / velocity(r)

# Function to compute travel time between two cities
def compute_travel_time(cityA, cityB):
    theta = compute_arc_angle(cityA, cityB)  # Arc angle between cities
    T, _ = spi.quad(travel_time_integrand, 0, theta, args=(theta,))
    return T

# Example cities (latitude, longitude)
city_A = (51.509865, -0.118092)  # London, UK
city_B = (40.712776, -74.005974) # New York, USA

# Compute travel time
T = compute_travel_time(city_A, city_B)

# Display results
print(f"Estimated travel time between {city_A} and {city_B} (hypocycloid path): {T / 60:.2f} minutes")




import numpy as np
import plotly.graph_objects as go

# Constants
R = 6371000  # Earth's radius in meters
b = R / 3  # Hypocycloid rolling circle radius

# Function to convert lat/lon to Cartesian coordinates
def latlon_to_cartesian(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])

# City coordinates (London, New York)
city_A = (51.509865, -0.118092)  # London, UK
city_B = (40.712776, -74.005974) # New York, USA

# Convert to Cartesian
A = latlon_to_cartesian(*city_A)
B = latlon_to_cartesian(*city_B)

# Compute the axis of rotation (normal to the plane of the two cities)
axis = np.cross(A, B)
axis /= np.linalg.norm(axis)  # Normalize the axis

# Compute arc angle between cities
theta_AB = np.arccos(np.dot(A, B) / (R**2))  # Angle in radians

# Generate the hypocycloid path inside the Earth
t_vals = np.linspace(0, theta_AB, 300)  # Correct parametric range

# Define hypocycloid in the plane of rotation
x_vals = (R - b) * np.cos(t_vals) + b * np.cos(((R - b) / b) * t_vals)
y_vals = (R - b) * np.sin(t_vals) - b * np.sin(((R - b) / b) * t_vals)
z_vals = np.zeros_like(x_vals)  # Initialize in xy-plane

# Create rotation matrix to align hypocycloid with the London-New York arc
def rotate_vector(v, axis, angle):
    """Rotate vector `v` around `axis` by `angle` using Rodrigues' rotation formula."""
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R_matrix @ v

# Rotate each point in the hypocycloid to align with the tunnel
rotated_path = np.array([rotate_vector(np.array([x, y, z]), axis, t) for x, y, z, t in zip(x_vals, y_vals, z_vals, t_vals)])

# Generate Earth's surface for visualization
phi, theta = np.mgrid[0:np.pi:30j, 0:2*np.pi:30j]
earth_x = R * np.sin(phi) * np.cos(theta)
earth_y = R * np.sin(phi) * np.sin(theta)
earth_z = R * np.cos(phi)

# Create Plotly figure
fig = go.Figure()

# Add Earth's surface
fig.add_trace(go.Surface(x=earth_x, y=earth_y, z=earth_z, opacity=0.5, colorscale='blues'))

# Add the corrected hypocycloid tunnel path
fig.add_trace(go.Scatter3d(
    x=rotated_path[:, 0], y=rotated_path[:, 1], z=rotated_path[:, 2],
    mode='lines', line=dict(color='red', width=5),
    name='Hypocycloid Tunnel'
))

# Add start and end points (cities)
fig.add_trace(go.Scatter3d(
    x=[A[0], B[0]], y=[A[1], B[1]], z=[A[2], B[2]],
    mode='markers', marker=dict(size=8, color=['green', 'blue']),
    name='Cities'
))

# Set layout
fig.update_layout(
    title="Fixed 3D Hypocycloid Underground Path (London to New York)",
    scene=dict(
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
        aspectmode="auto"
    )
)

# Show the plot
fig.show()



import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Constants
g = 9.81  # m/s² (gravity at Earth's surface)
R = 6371000  # m (Earth's radius)

# Function to compute arc angle
def compute_arc_angle(d):
    return (d * 1000) / R  # Convert km to meters, then compute arc angle in radians

# Compute best b dynamically (physics-based)
def compute_best_b(d):
    theta = compute_arc_angle(d)
    alpha = 0.5  # Scaling factor (can be tuned)
    b = alpha * R * np.sin(theta / 2)
    return min(R, b)  # Ensure b ≤ R

# Hypocycloid parametric equations
def x(t, theta, b):
    return (R - b) * np.cos(t) + b * np.cos(((R - b) / b) * t)

def y(t, theta, b):
    return (R - b) * np.sin(t) - b * np.sin(((R - b) / b) * t)

# First derivatives
def dx_dt(t, theta, b):
    return -(R - b) * np.sin(t) - b * ((R - b) / b) * np.sin(((R - b) / b) * t)

def dy_dt(t, theta, b):
    return (R - b) * np.cos(t) - b * ((R - b) / b) * np.cos(((R - b) / b) * t)

# Arc length element ds
def ds_dt(t, theta, b):
    return np.sqrt(dx_dt(t, theta, b)**2 + dy_dt(t, theta, b)**2)

# Velocity function using energy conservation
def velocity(r):
    return np.sqrt((g / R) * (R**2 - r**2))

# Travel time integral function
def travel_time_integrand(t, theta, b):
    r = np.sqrt(x(t, theta, b)**2 + y(t, theta, b)**2)
    return ds_dt(t, theta, b) / velocity(r)

# Compute travel time with best b
def compute_travel_time(d):
    theta = compute_arc_angle(d)
    b_optimal = compute_best_b(d)
    T, _ = spi.quad(travel_time_integrand, 0, theta, args=(theta, b_optimal))
    return T, b_optimal, theta

# Ask user for straight-line distance
d = float(input("🌍 Enter the straight-line distance between the two cities (in km): "))

# Compute best b, travel time, and angle
T, best_b, theta = compute_travel_time(d)

# Generate points for visualization
t_values = np.linspace(0, theta, 300)  # Smooth path
x_values = x(t_values, theta, best_b)
y_values = y(t_values, theta, best_b)

# Earth surface outline (for visualization)
earth_circle = np.linspace(0, 2 * np.pi, 300)
earth_x = R * np.cos(earth_circle)
earth_y = R * np.sin(earth_circle)

# Plot the tunnel path inside the Earth
plt.figure(figsize=(8, 8))
plt.plot(earth_x, earth_y, color="gray", linestyle="dashed", label="Earth's Surface")  # Earth's outline
plt.plot(x_values, y_values, label="Hypocycloid Tunnel Path", color="blue", linewidth=2)  # Tunnel path
plt.scatter([x_values[0], x_values[-1]], [y_values[0], y_values[-1]], color="red", s=100, label="Entry/Exit Points")  # Start & end

# Formatting the plot
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title(f"Hypocycloid Tunnel Path for {d} km Distance")
plt.legend()
plt.axis("equal")  # Keep Earth circular
plt.grid(True)
plt.show()

# Display results
print(f"\n🚄 Travel Time for Hypocycloid Tunnel over {d} km:")
print(f"✅ Best rolling circle radius (b): {best_b:.2f} meters")
print(f"🕒 Estimated travel time: {T / 60:.2f} minutes")





# With Louis' gravity model
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Constants
g = 9.81  # m/s² (gravity at Earth's surface)
R = 6371000  # m (Earth's radius)

# Louis' modified gravity function
def louis_gravity_function(r):
    """Modified gravity function based on Louis' model."""
    return g * (r / R)  # Gravity decreases linearly inside Earth

# Compute velocity using Louis' gravity function (from energy conservation)
def velocity_louis_model(r):
    """Velocity calculation using Louis' gravity model."""
    return np.sqrt(2 * (R - r) * louis_gravity_function(r))

# Compute acceleration using Louis' gravity function
def acceleration_louis_model(r):
    """Acceleration calculation using Louis' gravity model."""
    return louis_gravity_function(r)  # Directly from Louis' model

# Function to compute arc angle
def compute_arc_angle(d):
    return (d * 1000) / R  # Convert km to meters, then compute arc angle in radians

# Compute best b dynamically
def compute_best_b(d):
    theta = compute_arc_angle(d)
    alpha = 0.5  # Scaling factor
    b = alpha * R * np.sin(theta / 2)
    return min(R, b)  # Ensure b ≤ R

# Hypocycloid parametric equations
def x(t, theta, b):
    return (R - b) * np.cos(t) + b * np.cos(((R - b) / b) * t)

def y(t, theta, b):
    return (R - b) * np.sin(t) - b * np.sin(((R - b) / b) * t)

# First derivatives
def dx_dt(t, theta, b):
    return -(R - b) * np.sin(t) - b * ((R - b) / b) * np.sin(((R - b) / b) * t)

def dy_dt(t, theta, b):
    return (R - b) * np.cos(t) - b * ((R - b) / b) * np.cos(((R - b) / b) * t)

# Arc length element ds
def ds_dt(t, theta, b):
    return np.sqrt(dx_dt(t, theta, b)**2 + dy_dt(t, theta, b)**2)

# Travel time integral function using Louis' gravity model
def travel_time_integrand(t, theta, b):
    r = np.sqrt(x(t, theta, b)**2 + y(t, theta, b)**2)
    return ds_dt(t, theta, b) / velocity_louis_model(r)

# Compute travel time with Louis' gravity model
def compute_travel_time(d):
    theta = compute_arc_angle(d)
    b_optimal = compute_best_b(d)
    T, _ = spi.quad(travel_time_integrand, 0, theta, args=(theta, b_optimal))
    return T, b_optimal

# Ask user for straight-line distance
d = float(input("🌍 Enter the straight-line distance between the two cities (in km): "))

# Compute best b and travel time
T, best_b = compute_travel_time(d)

# Compute velocity and acceleration profiles
r_values = np.linspace(0, R, 100)  # Radius values inside Earth
velocities = velocity_louis_model(r_values)
accelerations = acceleration_louis_model(r_values)

# Display travel time results
print(f"\n🚄 Travel Time using Louis' Gravity Model over {d} km:")
print(f"✅ Best rolling circle radius (b): {best_b:.2f} meters")
print(f"🕒 Estimated travel time: {T / 60:.2f} minutes")

# Plot velocity profile using Louis' gravity function
plt.figure(figsize=(8, 6))
plt.plot(r_values / 1e3, velocities, label="Velocity (m/s)", color="blue")
plt.xlabel("Radius (km)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Profile using Louis' Gravity Model")
plt.legend()
plt.grid(True)
plt.show()

# Plot acceleration profile using Louis' gravity function
plt.figure(figsize=(8, 6))
plt.plot(r_values / 1e3, accelerations, label="Acceleration (m/s²)", color="red")
plt.xlabel("Radius (km)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration Profile using Louis' Gravity Model")
plt.legend()
plt.grid(True)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.optimize import minimize

def get_coordinates(city):
    """Fetch latitude and longitude of a city."""
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError(f"Could not find coordinates for {city}")

def chord_length(lat1, lon1, lat2, lon2, R=6371):
    """Compute the straight-line chord distance through the Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x1, y1, z1 = R * np.cos(lat1) * np.cos(lon1), R * np.cos(lat1) * np.sin(lon1), R * np.sin(lat1)
    x2, y2, z2 = R * np.cos(lat2) * np.cos(lon2), R * np.cos(lat2) * np.sin(lon2), R * np.sin(lat2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_hypocycloid(a, b, num_points=200):
    """Generate a simplified hypocycloid path with minimal cusps."""
    theta = np.linspace(0, np.pi, num_points)
    x = (a - b) * np.cos(theta) + b * np.cos((a - b) / b * theta)
    y = (a - b) * np.sin(theta) - b * np.sin((a - b) / b * theta)
    return x, y

def travel_time(a, b, g=9.81):
    """Compute travel time for a given a and b using arc length and velocity."""
    x, y = generate_hypocycloid(a, b)
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)  # Arc length elements
    v = np.sqrt(2 * g * (a - np.abs(y[:-1])))  # Velocity using correct energy conservation
    time = np.sum(ds / v) / 60  # Convert seconds to minutes
    return time

def optimize_hypocycloid(R, chord_dist):
    """Find optimal a and b to minimize travel time while avoiding complex paths."""
    def objective(params):
        a, b = params
        return travel_time(a, b)
    
    # Restrict a/b to small integer values to avoid high-order cusps
    valid_ratios = [2, 3, 4]
    best_a, best_b, best_time = None, None, float('inf')
    
    for ratio in valid_ratios:
        b = chord_dist / (2 * np.pi * ratio)
        a = R  # Keep a fixed to Earth's radius to maintain a reasonable path
        time = travel_time(a, b)
        if time < best_time:
            best_a, best_b, best_time = a, b, time
    
    return best_a, best_b, best_time

def plot_hypocycloid(x, y, city1, city2, R):
    """Plot the optimized hypocycloid path inside the Earth with proper scaling."""
    theta = np.linspace(0, 2 * np.pi, 300)
    earth_x = R * np.cos(theta)
    earth_y = R * np.sin(theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(earth_x, earth_y, label="Earth's Surface", color='gray', linestyle='dashed')
    plt.plot(x, y, label="Optimized Hypocycloid Path", color='blue')
    plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red', label=f"{city1} to {city2}")
    plt.xlabel("Distance (km)")
    plt.ylabel("Depth (km)")
    plt.title("Optimized Hypocycloid Tunnel Path Inside the Earth")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def main():
    city1 = input("Enter the first city: ")
    city2 = input("Enter the second city: ")
    
    lat1, lon1 = get_coordinates(city1)
    lat2, lon2 = get_coordinates(city2)
    
    chord_dist = chord_length(lat1, lon1, lat2, lon2)
    R = 6371  # Earth radius in km
    
    optimal_a, optimal_b, min_time = optimize_hypocycloid(R, chord_dist)
    print(f"Optimized a: {optimal_a:.2f} km, Optimized b: {optimal_b:.2f} km")
    print(f"Minimum travel time: {min_time:.2f} minutes")
    
    x, y = generate_hypocycloid(optimal_a, optimal_b)
    plot_hypocycloid(x, y, city1, city2, R)

if __name__ == "__main__":
    main()
