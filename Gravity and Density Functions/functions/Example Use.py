import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define DensityFunction class needed for unpickling
class DensityFunction:
    def __init__(self, input_function, surface_radius=6371000):
        self.input_function = input_function
        self.surface_radius = surface_radius
    
    def __call__(self, r):
        if np.isscalar(r):
            return self.input_function(r) if r <= self.surface_radius else 0
        else:
            return np.where(r <= self.surface_radius, self.input_function(r), 0)

# Load pre-computed models
with open('earth_density_fit.pkl', 'rb') as f:
    density_function = pickle.load(f)
    
with open('earth_gravity_fit.pkl', 'rb') as f:
    gravity_function = pickle.load(f)

# Create radius array for plotting
EARTH_RADIUS = 6371000
r = np.linspace(0, 1.5 * EARTH_RADIUS, 1000)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot density
ax1.plot(r/1000, [density_function(ri) for ri in r], 'b-', label='Density')
ax1.axvline(x=EARTH_RADIUS/1000, color='r', linestyle='--', label='Earth Surface')
ax1.set_xlabel('Radius (km)')
ax1.set_ylabel('Density (kg/m³)')
ax1.set_title('Earth Density Profile')
ax1.grid(True)
ax1.legend()

# Plot gravity
ax2.plot(r/1000, [gravity_function(ri) for ri in r], 'g-', label='Gravity')
ax2.axvline(x=EARTH_RADIUS/1000, color='r', linestyle='--', label='Earth Surface')
ax2.set_xlabel('Radius (km)')
ax2.set_ylabel('Gravitational Acceleration (m/s²)')
ax2.set_title('Earth Gravity Profile')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('earth_profiles.svg', format='svg', bbox_inches='tight')
plt.show()

# Example radius values
radii = {
    'center': 0,                  # Earth's center
    'surface': EARTH_RADIUS,      # Earth's surface
    'space': 1.2 * EARTH_RADIUS     # Twice Earth's radius
}

# Test density model
print("Density values:")
print(f"Center:  {density_function(radii['center']):.0f} kg/m³")    # ~13088 kg/m³
print(f"Surface: {density_function(radii['surface']):.0f} kg/m³")   # ~1020 kg/m³
print(f"Space:   {density_function(radii['space']):.0f} kg/m³")     # ~0 kg/m³

# Test gravity model
print("\nGravity values:")
print(f"Center:  {gravity_function(radii['center']):.2f} m/s²")     # 0 m/s²
print(f"Surface: {gravity_function(radii['surface']):.2f} m/s²")    # ~9.81 m/s²
print(f"Space:   {gravity_function(radii['space']):.2f} m/s²")      # ~2.45 m/s²