# Earth Density and Gravity Models

Pre-computed models for Earth's density and gravitational field stored as pickle files.

## Quick Start

```python
import pickle
import numpy as np

# Load pre-computed models
with open('earth_density_fit.pkl', 'rb') as f:
    density_function = pickle.load(f)
    
with open('earth_gravity_fit.pkl', 'rb') as f:
    gravity_function = pickle.load(f)

# Example radius values
EARTH_RADIUS = 6371000  # Earth's radius in meters
radii = {
    'center': 0,                  # Earth's center
    'surface': EARTH_RADIUS,      # Earth's surface
    'space': 2 * EARTH_RADIUS     # Twice Earth's radius
}

# Test density model
print("Density values:")
print(f"Center:  {density_function(radii['center']):.0f} kg/m³")    # ~13088 kg/m³
print(f"Surface: {density_function(radii['surface'])::.0f} kg/m³")   # ~1020 kg/m³
print(f"Space:   {density_function(radii['space'])::.0f} kg/m³")     # ~0 kg/m³

# Test gravity model
print("\nGravity values:")
print(f"Center:  {gravity_function(radii['center']):.2f} m/s²")     # 0 m/s²
print(f"Surface: {gravity_function(radii['surface']):.2f} m/s²")    # ~9.81 m/s²
print(f"Space:   {gravity_function(radii['space']):.2f} m/s²")      # ~2.45 m/s²
````

## DensityFunction Class

The density model uses a custom `DensityFunction` class that enforces the physical constraint of zero density beyond Earth's surface:

```python
class DensityFunction:
    def __init__(self, input_function, surface_radius=6371000):
        self.input_function = input_function
        self.surface_radius = surface_radius
    
    def __call__(self, r):
        if np.isscalar(r):
            return self.input_function(r) if r <= self.surface_radius else 0
        else:
            return np.where(r <= self.surface_radius, self.input_function(r), 0)