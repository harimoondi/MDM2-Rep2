import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 6371e3  # Earth's radius in m
g = 9.81  # Surface gravity in m/s^2

# Compute travel time for point-source gravity model (Keplerian-like motion)
T_full = 2 * np.pi * np.sqrt(R / g)  # Full oscillation period
T_half = T_full / 2  # Time for one-way trip

# Display travel time
T_half_minutes = T_half / 60  # Convert to minutes
T_half_minutes
print(T_half_minutes)

# Function to compute travel time for a chord-like tunnel
def travel_time_chord(distance):
    """Compute travel time for a chord-like tunnel using the hypocycloid model."""
    b = distance / (2 * np.pi)  # Inner rolling circle radius from the hypocycloid equation
    time = np.sqrt(R * (R - b)) / (g * b) * 2 * np.pi * b / R  # Travel time equation
    return time / 60  # Convert to minutes

# Example distances (in m) for major city pairs
distances_km = {
    "New York to Los Angeles": 4800,
    "London to Tokyo": 9600,
    "Sydney to Cape Town": 11600,
    "Paris to Beijing": 8200
}

# Convert distances to meters and compute travel times
travel_times = {city_pair: travel_time_chord(dist * 1e3) for city_pair, dist in distances_km.items()}

import pandas as pd
df = pd.DataFrame(travel_times.items(), columns=["City Pair", "Travel Time (mins)"])
print(df.to_string(index=False))  # Prints the DataFrame in a readable format

