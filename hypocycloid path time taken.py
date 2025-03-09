import numpy as np
import scipy.integrate as spi

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
    return T, b_optimal

# Ask user for straight-line distance
d = float(input("🌍 Enter the straight-line distance between the two cities (in km): "))

# Compute best b and travel time
T, best_b = compute_travel_time(d)

# Display results
print(f"\n🚄 Travel Time for Hypocycloid Tunnel over {d} km:")
print(f"✅ Best rolling circle radius (b): {best_b:.2f} meters")
print(f"🕒 Estimated travel time: {T / 60:.2f} minutes")
