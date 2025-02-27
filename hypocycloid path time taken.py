import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
R = 6371e3  # Earth's radius in m
g = 9.81  # Surface gravity in m/s^2

# Function to compute travel time for a chord-like tunnel
def travel_time_chord(distance):
    """Compute travel time for a chord-like tunnel using the hypocycloid model."""
    b = distance / (2 * np.pi)  # Inner rolling circle radius from the hypocycloid equation
    if b >= R:
        raise ValueError("Distance is too large for a chord-like tunnel Use the straight-line tunnel instead.")
    # time calculation
    time = (2 * np.pi / R) * np.sqrt(R*(R-b)/(g*b))
    return time * 60  # Convert time to minutes

# Test cases 
distances_km = {
    "New York to Los Angeles": 4800,
    "London to Tokyo": 9600,
    "Sydney to Cape Town": 11600,
    "Paris to Beijing": 8200
}

# Convert distances to meters and compute travel times
travel_times = {}
for city_pair, dist in distances_km.items():
    try:
        travel_times[city_pair] = travel_time_chord(dist * 1e3)
    except ValueError as e:
        travel_times[city_pair] = str(e)  # Store the error message instead of the time

# Results
Time_result = pd.DataFrame(travel_times.items(), columns=["City Pair", "Travel Time (mins) or Error"])

# Print results
print(Time_result.to_string(index=False))
