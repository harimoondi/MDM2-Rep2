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