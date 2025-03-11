'''
This script is used to attempt to validate the model. We know that the brachistochrone curve
is the fastest path through a uniform gravity field. So we set out a unifrom gravitational field ( 9.81 everywhere),
and plot the brachichrone curve, then make an initial guess near this curve and the minimizer should return something
very close to the theoretical brachistochrone curve. MSE is used to calculate the error between the two curves. (should be small)
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.constants as const
import pickle
from scipy.interpolate import interp1d
import time

# Load models once
try:
    with open("Gravity and Density Functions/functions/earth_gravity_fit.pkl", "rb") as f:
        prem_gravity_model = pickle.load(f)
    print("PREM gravity model loaded successfully")
except Exception as e:
    print(f"Error loading gravity model: {e}")
    prem_gravity_model = None

# Constants
r_earth = 6371000  # Earth radius in meters
G = const.G
M = 5.972e24
R = 6371000
# Coordinates
init_lat = 90.0  # North Pole
init_lon = 0.0  # Prime Meridian
final_lat = 0.0  # Equator
final_lon = 90.0  # 90 degrees East

# Convert to radians
theta_init = np.radians(90 - init_lat)
phi_init = np.radians(init_lon)
theta_end = np.radians(90 - final_lat)
phi_end = np.radians(final_lon)

def central_angle_calculation(theta_init, phi_init, theta_end, phi_end):
    return np.arccos(np.sin(theta_init) * np.sin(theta_end) +
                     np.cos(theta_init) * np.cos(theta_end) * np.cos(phi_end - phi_init))

def uniform_gravity_function(r):
    return np.where(r == 0, 0, G * M / R ** 2)

def uniform_density_function(r):
    return ((G * M / R**3)*r)

def prem_gravity_function(r):
    """Use the pre-loaded PREM gravity model"""
    if prem_gravity_model is not None:
        return prem_gravity_model(r)
    else:
        # Fallback if loading failed
        return uniform_gravity_function(r)

def velocity_uniform_gravity(r_values):
    """Velocity using constant gravity"""
    g = G * M / (R**2)  # Constant gravity
    potential_energy = g * (r_values - R)
    kinetic_energy = -potential_energy
    kinetic_energy = np.maximum(kinetic_energy, 0)
    velocities = np.sqrt(2 * kinetic_energy)
    return velocities

def velocity_uniform_density(r_values):
    """Velocity using gravity that increases linearly with depth (uniform density)"""
    potential_energy = np.zeros_like(r_values)
    
    for i in range(len(r_values)):
        segment_r = np.linspace(r_earth, r_values[i], 100)
        segment_g = uniform_density_function(segment_r)
        dr = (segment_r[1] - segment_r[0])
        potential_energy[i] = np.sum(segment_g * dr)
    
    kinetic_energy = -potential_energy
    kinetic_energy = np.maximum(kinetic_energy, 0)
    velocities = np.sqrt(2 * kinetic_energy)
    return velocities

def velocity_prem(r_values):
    """Velocity using PREM gravity model"""
    potential_energy = np.zeros_like(r_values)
    
    # Create more efficient sampling
    for i in range(len(r_values)):
        if r_values[i] >= r_earth:
            continue  # No potential energy difference at or above surface
            
        # Use fewer points for better performance
        n_points = 100
        segment_r = np.linspace(r_earth, r_values[i], n_points)
        segment_g = np.array([prem_gravity_function(r) for r in segment_r])
        
        # Calculate potential energy through integration
        potential_energy[i] = -np.trapz(segment_g, segment_r)
    
    kinetic_energy = -potential_energy
    kinetic_energy = np.maximum(kinetic_energy, 0)
    velocities = np.sqrt(2 * kinetic_energy)
    
    return velocities

def velocity_improved(r_values):
    """Calculate velocity at each point using conservation of energy with uniform gravity"""
    # For uniform gravity, use constant g = G*M/R²
    g = G * M / (R**2)  # Constant gravity
    
    # Calculate potential energy (relative to surface)
    potential_energy = g * (r_values - R)
    
    # Energy conservation: E = 0 at the starting point
    kinetic_energy = -potential_energy
    
    # Ensure no negative kinetic energy
    kinetic_energy = np.maximum(kinetic_energy, 0)
    
    # v = sqrt(2*KE)
    velocities = np.sqrt(2 * kinetic_energy)
    
    return velocities

def arc_length(r, dr, dalpha):
    return np.sqrt(dr ** 2 + r ** 2 * dalpha ** 2)

def time_integral(r_values, alpha_values):
    v_values = velocity_improved(r_values)
    dr_values = np.diff(r_values)
    dalpha_values = np.diff(alpha_values)
    time_values = np.zeros(len(dr_values))

    for i in range(len(dr_values)):
        # Use average velocity between points
        v_avg = 0.5 * (v_values[i] + v_values[i+1])
        
        # Small epsilon to prevent division by zero
        if v_avg < 1e-10:
            v_avg = 1e-10
            
        ds = arc_length(r_values[i], dr_values[i], dalpha_values[i])
        time_values[i] = ds / v_avg

    total_time = np.sum(time_values)
    return total_time

def objective(params, alpha_values):
    r_values = params
    r_values[0] = r_values[-1] = r_earth
    total_time = time_integral(r_values, alpha_values)
    return total_time

def straight_line_initial_guess(R, num_points=200):
    """Create a straight line tunnel initial guess (definitely not optimal)"""
    # Straight line would be a diagonal across the quarter circle
    r_initial = np.ones(num_points) * R
    
    # Make middle points dip a bit to give some variation
    middle_idx = num_points // 2
    dip_factor = 0.7  # How much to dip (0.7 means 70% of Earth radius)
    
    # Apply a simple parabolic dip
    x = np.linspace(-1, 1, num_points)
    dip_profile = 1 - (1 - dip_factor) * (1 - x**2)
    
    r_initial = r_initial * dip_profile
    
    # Ensure boundary conditions
    r_initial[0] = r_initial[-1] = R
    
    return r_initial

def theoretical_brachistochrone(R):
    t_values = np.linspace(0, np.pi, 200)
    x = R * (t_values - np.sin(t_values))
    y = R * (1 - np.cos(t_values))

    magnitude = np.array(np.sqrt(x ** 2 + y ** 2))
    # Scale Magnitude
    scale_mag = R / magnitude[-1]
    magnitude = magnitude * scale_mag

    x_scale = R / x[-1]
    y_scale = R / y[-1]
    x = x * x_scale
    y = y * y_scale
    y = -y
    y = y + R
    return x, y, magnitude

def chebyshev_nodes(a, b, n):
    """Generate Chebyshev nodes in the interval [a, b]
    
    This creates nodes that are more densely distributed near the edges
    of the interval, which is useful for better approximation at boundaries.
    """
    k = np.arange(n)
    nodes = (a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) * np.pi / (2 * n))
    # Return sorted nodes (Chebyshev nodes are naturally ordered from b to a)
    return np.sort(nodes)

def optimize_path_normalized(r_init, alpha_values):
    """Optimization with normalized variables to improve numerical stability"""
    n_points = len(r_init)
    
    # Normalize initial guess by Earth's radius
    r_init_norm = r_init / r_earth
    
    # Normalized bounds between 0.1 and 1.0
    bounds = [(0.1, 1.0) for _ in range(n_points)]
    bounds[0] = bounds[-1] = (1.0, 1.0)  # Fixed endpoints
    
    def normalized_objective(x_norm):
        # Convert normalized values back to real values for calculation
        x_real = x_norm * r_earth
        x_real[0] = x_real[-1] = r_earth  # Ensure endpoints are fixed
        
        # Calculate time using real values
        return time_integral(x_real, alpha_values)
    
    # First try direct optimization
    result = minimize(
        normalized_objective,
        r_init_norm,
        method='SLSQP',
        bounds=bounds,
        options={
            'ftol': 1e-8,
            'maxiter': 1000,
            'disp': True
        }
    )
    
    # Convert back to real values
    optimized_r_values = result.x * r_earth
    optimized_r_values[0] = optimized_r_values[-1] = r_earth
    
    return optimized_r_values

def optimize_with_gravity_model(r_init, alpha_values, velocity_function):
    """Run optimization with the specified gravity function"""
    
    # Store the original velocity function
    original_velocity = globals()['velocity_improved']
    
    # Replace with our selected function
    globals()['velocity_improved'] = velocity_function
    
    # Run optimization
    optimized_r = optimize_path_normalized(r_init, alpha_values)
    
    # Restore original function
    globals()['velocity_improved'] = original_velocity
    
    return optimized_r

def setup_earth_plot(title, figsize=(10, 8)):
    """Helper function to set up a plot with Earth's surface"""
    plt.figure(figsize=figsize)
    
    # Plot Earth's surface
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = r_earth * np.cos(theta)
    y_circle = r_earth * np.sin(theta)
    plt.plot(x_circle, y_circle, 'k-', linewidth=1, label='Earth Surface')
    
    # Mark the start and end points
    plt.plot(0, r_earth, 'ko', markersize=8, label='Start')
    plt.plot(r_earth, 0, 'ko', markersize=8, label='End')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    
    # Zoom in to show the details better
    zoom_factor = 0.2
    plt.xlim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
    plt.ylim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
    
    return plt.gca()  # Return the current axis for further customization

def plot_uniform_gravity_model():
    """Plot the brachistochrone curve for uniform gravity model"""
    print("Generating uniform gravity model plot...")
    
    # Setup required variables
    num_points = 200
    r_initial = straight_line_initial_guess(R, num_points)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Set to uniform gravity model
        globals()['velocity_improved'] = velocity_uniform_gravity
        
        # Optimize path
        print("Optimizing uniform gravity path...")
        optimized_r = optimize_path_normalized(r_initial, alpha_values)
        travel_time = time_integral(optimized_r, alpha_values) / 60
        print(f"Travel time with uniform gravity: {travel_time:.2f} minutes")
        
        # Create plot
        ax = setup_earth_plot('Brachistochrone Path with Uniform Gravity Model')
        
        # Plot the optimized path
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                'r-', linewidth=3, label=f'Uniform Gravity (Travel time: {travel_time:.2f} min)')
        
        # Add theoretical brachistochrone
        x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
        plt.plot(x_theoretical, y_theoretical, 'm--', linewidth=1.5, label='Theoretical Brachistochrone')
        
        plt.legend(loc='upper right')
        plt.savefig('uniform_gravity_model.png')
        plt.show()
        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

def plot_uniform_density_model():
    """Plot the brachistochrone curve for uniform density model"""
    print("Generating uniform density model plot...")
    
    # Setup required variables
    num_points = 200
    r_initial = straight_line_initial_guess(R, num_points)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Set to uniform density model
        globals()['velocity_improved'] = velocity_uniform_density
        
        # Optimize path
        print("Optimizing uniform density path...")
        optimized_r = optimize_path_normalized(r_initial, alpha_values)
        travel_time = time_integral(optimized_r, alpha_values) / 60
        print(f"Travel time with uniform density: {travel_time:.2f} minutes")
        
        # Create plot
        ax = setup_earth_plot('Brachistochrone Path with Uniform Density Model')
        
        # Plot the optimized path
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                'g-', linewidth=3, label=f'Uniform Density (Travel time: {travel_time:.2f} min)')
        
        plt.legend(loc='upper right')
        plt.savefig('uniform_density_model.png')
        plt.show()
        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

def plot_prem_model():
    """Plot the brachistochrone curve for PREM gravity model"""
    print("Generating PREM model plot...")
    
    # Setup required variables
    num_points = 200
    r_initial = straight_line_initial_guess(R, num_points)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Set to PREM model
        globals()['velocity_improved'] = velocity_prem
        
        # Optimize path
        print("Optimizing PREM model path...")
        optimized_r = optimize_path_normalized(r_initial, alpha_values)
        travel_time = time_integral(optimized_r, alpha_values) / 60
        print(f"Travel time with PREM model: {travel_time:.2f} minutes")
        
        # Create plot
        ax = setup_earth_plot('Brachistochrone Path with PREM Gravity Model')
        
        # Plot the optimized path
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                'b-', linewidth=3, label=f'PREM Model (Travel time: {travel_time:.2f} min)')
        
        plt.legend(loc='upper right')
        plt.savefig('prem_model.png')
        plt.show()
        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

def plot_simple_gravity_comparison():
    print("Starting gravity model comparison...")
    
    # Use the same point count as the initial optimization
    num_points = 200
    print(f"Generating initial guess with {num_points} points...")
    r_initial = straight_line_initial_guess(R, num_points)
    
    # Use the same alpha distribution as the initial optimization
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    print(f"Generated Chebyshev nodes for angular distribution")
    
    # Save original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        print("\n===== Uniform Gravity Model =====")
        print("Optimizing uniform gravity model...")
        # Use the same optimization method as the initial plot
        globals()['velocity_improved'] = velocity_uniform_gravity
        start_time = time.time()
        optimized_r_uniform_gravity = optimize_path_normalized(r_initial, alpha_values)
        time_uniform_gravity = time_integral(optimized_r_uniform_gravity, alpha_values)/60
        print(f"Optimization completed in {(time.time() - start_time):.2f} seconds")
        print(f"Travel time with uniform gravity: {time_uniform_gravity:.2f} minutes")
        
        print("\n===== Uniform Density Model =====")
        print("Optimizing uniform density model...")
        globals()['velocity_improved'] = velocity_uniform_density
        start_time = time.time()
        optimized_r_uniform_density = optimize_path_normalized(r_initial, alpha_values)
        time_uniform_density = time_integral(optimized_r_uniform_density, alpha_values)/60
        print(f"Optimization completed in {(time.time() - start_time):.2f} seconds")
        print(f"Travel time with uniform density: {time_uniform_density:.2f} minutes")
        
        print("\n===== PREM Model =====")
        print("Optimizing PREM model...")
        globals()['velocity_improved'] = velocity_prem
        start_time = time.time()
        optimized_r_prem = optimize_path_normalized(r_initial, alpha_values)
        time_prem = time_integral(optimized_r_prem, alpha_values)/60
        print(f"Optimization completed in {(time.time() - start_time):.2f} seconds")
        print(f"Travel time with PREM model: {time_prem:.2f} minutes")
        
        # Create plot
        print("\nGenerating comparison plot...")
        plt.figure(figsize=(14, 10))
        
        # Plot Earth's surface
        print("Drawing Earth's surface...")
        theta = np.linspace(0, 2 * np.pi, 1000)
        x_circle = r_earth * np.cos(theta)
        y_circle = r_earth * np.sin(theta)
        plt.plot(x_circle, y_circle, 'k-', linewidth=1, label='Earth Surface')
        
        # Plot the optimized paths
        print("Adding optimized paths to plot...")
        plt.plot(optimized_r_uniform_gravity * np.cos(alpha_values), 
                optimized_r_uniform_gravity * np.sin(alpha_values), 
                'r-', linewidth=2, label='Uniform Gravity')
                
        plt.plot(optimized_r_uniform_density * np.cos(alpha_values), 
                optimized_r_uniform_density * np.sin(alpha_values), 
                'g-', linewidth=2, label='Uniform Density')
                
        plt.plot(optimized_r_prem * np.cos(alpha_values), 
                optimized_r_prem * np.sin(alpha_values), 
                'b-', linewidth=2, label='PREM Model')
        
        # Plot theoretical brachistochrone for reference
        print("Adding theoretical brachistochrone curve...")
        x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
        plt.plot(x_theoretical, y_theoretical, 'm--', linewidth=1, label='Theoretical Brachistochrone')
        
        # Mark the start and end points
        plt.plot(0, r_earth, 'ko', markersize=8)
        plt.plot(r_earth, 0, 'ko', markersize=8)
        
        # Zoom in to show the differences better
        zoom_factor = 0.2
        plt.xlim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
        plt.ylim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
        
        plt.title('Comparison of Brachistochrone Paths for Different Gravity Models')
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='upper right')
        print("Saving plot to 'gravity_model_comparison.png'...")
        plt.savefig('gravity_model_comparison.png')
        print("Displaying plot...")
        plt.show()
        
        # Print travel times
        print(f"\nTravel times summary (minutes):")
        print(f"Uniform Gravity: {time_uniform_gravity:.2f}")
        print(f"Uniform Density: {time_uniform_density:.2f}")
        print(f"PREM Model: {time_prem:.2f}")
        print("\nComparison complete!")
        
    except Exception as e:
        print(f"Error during gravity model comparison: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always restore original velocity function
        globals()['velocity_improved'] = original_velocity
        print("Restored original velocity function")

# Update the main function to include options for individual plots
if __name__ == "__main__":
    
    plot_uniform_gravity_model()

    ## plot_uniform_density_model()

    ## plot_prem_model()
    
    ## plot_simple_gravity_comparison()

