import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.interpolate import CubicSpline

# Constants
R_earth = 6371000  # Earth radius in meters
G = const.G
M = 5.972e24
R = 6371000


# Linear gravitational
def linear_I(r):
    return (G * M / R**3) * r

def PREM_I(r):
    file_path = "Gravity and Density Functions/functions/earth_gravity_fit.pkl"
    with open(file_path, "rb") as f:
        non_gravity_function = pickle.load(f)
    return non_gravity_function(r)


def calculate_dr_dh(r, R_d):
    # This function calculates dr/dh for the brachistochrone path
    try:
        # Handle numerical issues with the square root
        term = (R_d/r)**2 - 1
        if term <= 0:  # This happens at or beyond the deepest point
            return 0.0  # Return zero at the bottom of the path
        
        dr_dh = r / np.sqrt(term)
        # Limit extreme values
        return min(dr_dh, 1e6)  # Cap at a reasonable value
    except:
        # Fallback for any errors
        return 0.0


def calculate_path(R_d, dh=0.001, max_iter=50000):
    """
    Calculate path through Earth with maximum depth R_d
    dh: step size for angle integration
    max_iter: maximum iterations to prevent infinite loops
    """
    # Starting conditions (slightly above the deepest point)
    r = R_d * 1.001  # Start slightly above maximum depth to avoid singularity
    h = 0.0  # Starting angle

    # Arrays to store path
    r_values = [r]
    h_values = [h]
    
    iterations = 0
    surface_reached = False

    # Integrate outward to reach Earth's surface
    while iterations < max_iter:
        dr_dh_val = calculate_dr_dh(r, R_d)
        
        # Euler integration step
        r_new = r + dr_dh_val * dh
        h_new = h + dh
        
        # Check if we've reached or passed the surface
        if r_new >= R_earth:
            # Interpolate to find where the path crosses the surface
            t = (R_earth - r) / (r_new - r)
            h_surface = h + t * (h_new - h)
            
            # Add the surface point
            r_values.append(R_earth)
            h_values.append(h_surface)
            
            surface_reached = True
            break
        
        # Store new values
        r_values.append(r_new)
        h_values.append(h_new)
        
        # Update current position
        r = r_new
        h = h_new
        iterations += 1
    
    # If we didn't reach the surface after all iterations
    if not surface_reached:
        print(f"Warning: Integration stopped after {iterations} steps")
        print(f"Last point: r={r/1000:.1f} km (vs Earth radius={R_earth/1000:.1f} km)")
        print("Forcing path to reach surface by linear extrapolation")
        
        # Force the last point to be on the surface 
        r_values.append(R_earth)
        h_values.append(h_values[-1] + 0.01)  # Add a small angle increment
    
    # Now generate the symmetric part
    h_max = h_values[-1]  # The angle at the surface 
    r_values_2nd_half = r_values[:-1][::-1]  # Reverse and exclude the last point
    h_values_2nd_half = [2*h_max - h for h in h_values[:-1][::-1]]  # Reflect and exclude the last point
    
    # Combine both halves
    full_r_values = np.array(r_values + r_values_2nd_half)
    full_h_values = np.array(h_values + h_values_2nd_half)
    
    return full_r_values, full_h_values

def velocity(r_values, gravity_function=PREM_I):
    # Calculate potential energy at each point relative to the surface
    potential_energy = np.zeros_like(r_values, dtype=float)
    
    # Use fewer points for better performance
    n_points = 50
    
    # For all points, calculate potential from surface
    for i in range(len(r_values)):
        if r_values[i] < R_earth - 1:  # Only for points below surface (with small margin)
            # Sample points between surface and current radius
            sample_r = np.linspace(R_earth, r_values[i], n_points)
            # Get gravity at each point
            sample_g = np.array([gravity_function(r) for r in sample_r])
            # Integrate to get potential
            potential_energy[i] = -np.trapz(sample_g, sample_r)
    
    # Convert potential energy to velocity using conservation of energy
    # v^2/2 = -(PE - PE_surface), where PE_surface = 0
    v_squared = -2 * potential_energy
    
    # Ensure no negative values due to numerical errors
    v_squared = np.maximum(v_squared, 0)
    
    # Add a small buffer to avoid division by zero
    velocities = np.sqrt(v_squared)
    
    # Set a minimum velocity to avoid division by zero 
    velocities = np.maximum(velocities, 0.1)
    
    return velocities

def velocity_optimized(r_values, gravity_function=PREM_I):
    print(f"Calculating velocities for {len(r_values)} points using optimized method...")
    
    # Pre-calculate gravity values at multiple depths
    # This creates a lookup table to interpolate from, saving computation time
    depths = np.linspace(0, R_earth, 1000)
    radii = R_earth - depths
    gravity_values = np.array([gravity_function(r) for r in radii])
    
    # Create an interpolation function
    from scipy.interpolate import interp1d
    gravity_interp = interp1d(radii, gravity_values, bounds_error=False, fill_value=(gravity_values[0], gravity_values[-1]))
    
    # Calculate potential energy at each point
    potential_energy = np.zeros_like(r_values, dtype=float)
    
    # Show progress at intervals
    update_interval = max(1, len(r_values) // 20)
    
    # For all points, calculate potential from surface
    for i in range(len(r_values)):
        if i % update_interval == 0:
            print(f"  Progress: {i}/{len(r_values)} points ({i/len(r_values)*100:.1f}%)")
            
        if r_values[i] < R_earth - 1:  # Only for points below surface
            # Sample points between surface and current radius
            sample_r = np.linspace(R_earth, r_values[i], 200)
            # Use interpolated gravity values
            sample_g = gravity_interp(sample_r)
            # Integrate to get potential
            potential_energy[i] = -np.trapz(sample_g, sample_r)
    
    print("Converting potential energy to velocity...")
    v_squared = -2 * potential_energy
    v_squared = np.maximum(v_squared, 0)
    velocities = np.sqrt(v_squared)
    velocities = np.maximum(velocities, 0.1)
    
    print("Velocity calculation complete.")
    return velocities

def travel_time(r_values, h_values, velocities):
    """Calculate travel time along the path."""
    # Convert inputs to numpy arrays if they aren't already
    r_values = np.asarray(r_values)
    h_values = np.asarray(h_values)
    velocities = np.asarray(velocities)
    
    # Initialize time array
    times = np.zeros_like(r_values)
    
    for i in range(1, len(r_values)):
        r1, r2 = r_values[i - 1], r_values[i]
        h1, h2 = h_values[i - 1], h_values[i]
        
        # Get average velocity for this segment (avoid division by zero)
        v = max(0.001, (velocities[i-1] + velocities[i]) / 2)
        
        # Calculate path length for this segment
        dr = r2 - r1
        dh = h2 - h1
        ds = np.sqrt(dr**2 + (r1 * dh)**2)  # Arc length in polar coordinates
        
        # Time to traverse this segment
        dt = ds / v
        times[i] = times[i-1] + dt
    
    return times  # This will be a numpy array

def coords_to_radians(init_lat,init_lon, final_lat, final_lon):
    theta_init = np.radians(90 - init_lat)
    phi_init = np.radians(init_lon)
    theta_end = np.radians(90 - final_lat)
    phi_end = np.radians(final_lon)

    return theta_init, phi_init, theta_end, phi_end


def main():
    # Set maximum depth as fraction of Earth radius
    R_d = 0.5 * R_earth  # Use a larger fraction of radius (closer to center)
    
    # Integration step size - use much smaller step for better integration
    dh = 0.0001
    
    # Calculate path
    print("Calculating path...")
    r_values, h_values = calculate_path(R_d, dh)
    print(f"Path calculated: {len(r_values)} points")
    print(f"Min radius: {min(r_values)/1000:.2f} km, Max radius: {max(r_values)/1000:.2f} km")
    print(f"Angle range: {min(h_values):.4f} to {max(h_values):.4f} radians")
    
    # Ensure r_values are within reasonable bounds
    r_values = np.minimum(r_values, R_earth * 1.01)  # Cap at slightly above Earth radius
    r_values = np.maximum(r_values, R_d * 0.99)      # Ensure lower bound
    
    # Calculate velocities - try using regular version if optimized is too slow
    print("Calculating velocities...")
    # Use regular version with fewer points if you prefer
    velocities = velocity(r_values[::20])  # Sample every 20th point
    r_sampled = r_values[::20]
    h_sampled = h_values[::20]
    
    print(f"Max velocity: {np.max(velocities):.2f} m/s")
    
    # Calculate travel time
    print("Calculating travel time...")
    times = travel_time(r_sampled, h_sampled, velocities)
    
    # Convert to cartesian coordinates for plotting
    x_values = r_sampled * np.cos(h_sampled)
    y_values = r_sampled * np.sin(h_sampled)
    
    # Plot path
    plt.figure(figsize=(12, 12))
    plt.plot(x_values/1000, y_values/1000, 'b-', linewidth=2)
    
    # Plot Earth circle
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(R_earth/1000 * np.cos(theta), R_earth/1000 * np.sin(theta), 'k-', linewidth=1)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'Brachistochrone Path through Earth (Max depth: {(R_earth-R_d)/1000:.0f} km)')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.savefig('earth_path.png')
    
    # Add velocity plot
    plt.figure(figsize=(10, 6))
    plt.plot(times/60, velocities, 'g-', linewidth=2)
    plt.grid(True)
    plt.title('Velocity vs Time')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Velocity (m/s)')
    plt.savefig('velocity_profile.png')
    
    if np.isnan(times[-1]):
        print("Error: Travel time calculation failed (NaN result)")
    else:
        print(f"Total travel time: {times[-1] / 60:.2f} minutes")

main()
