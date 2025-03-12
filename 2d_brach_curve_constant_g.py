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
    """Velocity using gravity that increases linearly with depth (uniform density)"""
    potential_energy = np.zeros_like(r_values)
    
    for i in range(len(r_values)):
        segment_r = np.linspace(r_earth, r_values[i], 100)
        segment_g = prem_gravity_function(segment_r)
        dr = (segment_r[1] - segment_r[0])
        potential_energy[i] = np.sum(segment_g * dr)
    
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

def brachistochrone_initial_guess(R, num_points=200):
    """Create an initial guess based on the theoretical brachistochrone curve"""
    # Get the theoretical brachistochrone curve
    x_brach, y_brach, _ = theoretical_brachistochrone(R)
    
    # Create interpolation function to sample at desired resolution
    from scipy.interpolate import interp1d
    
    # Compute angle values for each point on the theoretical curve
    alpha_brach = np.arctan2(y_brach, x_brach)
    
    # Sort to ensure angles are monotonically increasing
    indices = np.argsort(alpha_brach)
    alpha_sorted = alpha_brach[indices]
    r_sorted = np.sqrt(x_brach[indices]**2 + y_brach[indices]**2)
    
    # Remove duplicate angles if any
    unique_indices = np.concatenate(([True], np.diff(alpha_sorted) > 0))
    alpha_unique = alpha_sorted[unique_indices]
    r_unique = r_sorted[unique_indices]
    
    # Create interpolation function
    interp_func = interp1d(alpha_unique, r_unique, bounds_error=False, fill_value="extrapolate")
    
    # Generate evenly spaced angles for sampling
    central_angle = np.pi/2  # Quarter circle from (0,R) to (R,0)
    alpha_values = np.linspace(0, central_angle, num_points)
    
    # Sample r values at these angles
    r_initial = interp_func(alpha_values)
    
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
            'maxiter': 100,
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
    plt.plot(0, r_earth, 'ko', color='r', markersize=8, label='Start')
    plt.plot(r_earth, 0, 'ko', markersize=8, label='End')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    
    # Zoom in to show the details better
    zoom_factor = 0.075
    plt.xlim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
    plt.ylim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
    
    return plt.gca()  # Return the current axis for further customization

def plot_uniform_gravity_model():
    """Plot the brachistochrone curve for uniform gravity model"""
    print("Generating uniform gravity model plot...")
    
    # Setup required variables
    num_points = 50
    r_initial = brachistochrone_initial_guess(R, num_points)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Set to uniform gravity model
        globals()['velocity_improved'] = velocity_uniform_gravity
        
        # Plot initial guess and Earth
        ax = setup_earth_plot('Brachistochrone Path with Uniform Gravity Model')
        plt.plot(r_initial * np.cos(alpha_values), 
                r_initial * np.sin(alpha_values), 
                'k--', linewidth=1.5, label='Initial Guess')

        # Optimize path
        print("Optimizing uniform gravity path...")
        optimized_r = optimize_path_normalized(r_initial, alpha_values)
        travel_time = time_integral(optimized_r, alpha_values) / 60
        print(f"Travel time with uniform gravity: {travel_time:.2f} minutes")
        
        # Plot the optimized path
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                'r-', linewidth=3, label=f'Uniform Gravity (Travel time: {travel_time:.2f} min)')
        
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                 'm--', linewidth=1.5, label='Theoretical Brachistochrone')

        plt.legend(loc='upper right')
        plt.savefig('uniform_gravity_model.png')
        plt.show()

        # Print values for analysis
        print("\nOptimized radius values (uniform gravity):")
        print(f"Min: {np.min(optimized_r):.0f} m, Max: {np.max(optimized_r):.0f} m")
        print(f"First few values: {optimized_r[:5]}")
        print(f"Middle values: {optimized_r[len(optimized_r)//2-2:len(optimized_r)//2+3]}")
        print(f"Last few values: {optimized_r[-5:]}")
        
        # Check symmetry
        mid_point = len(optimized_r) // 2
        left_half = optimized_r[:mid_point]
        right_half = optimized_r[mid_point:][::-1]  # Reverse the right half
        
        # If lengths are different (odd number of points), truncate
        min_length = min(len(left_half), len(right_half))
        left_half = left_half[:min_length]
        right_half = right_half[:min_length]
        
        symmetry_diff = np.abs(left_half - right_half)
        print(f"\nSymmetry check - Average difference: {np.mean(symmetry_diff):.2f} m")
        print(f"Max symmetry difference: {np.max(symmetry_diff):.2f} m")
        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

def plot_uniform_density_model():
    """Plot the brachistochrone curve for uniform density model"""
    print("Generating uniform density model plot...")
    
    # Setup required variables
    num_points = 50
    r_initial = brachistochrone_initial_guess(R, num_points)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Set to uniform density model
        globals()['velocity_improved'] = velocity_uniform_density
        
        # Plot initial guess and Earth
        ax = setup_earth_plot('Brachistochrone Path with Uniform Density Model')
        plt.plot(r_initial * np.cos(alpha_values), 
                r_initial * np.sin(alpha_values), 
                'k--', linewidth=1.5, label='Initial Guess')

        # Optimize path
        print("Optimizing uniform density path...")
        optimized_r = optimize_path_normalized(r_initial, alpha_values)
        travel_time = time_integral(optimized_r, alpha_values) / 60
        print(f"Travel time with uniform density: {travel_time:.2f} minutes")
        
        # Plot the optimized path
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                'g-', linewidth=3, label=f'Uniform Density (Travel time: {travel_time:.2f} min)')
        
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                 'm--', linewidth=1.5, label='Theoretical Brachistochrone')

        plt.legend(loc='upper right')
        plt.savefig('uniform_density_model.png')
        plt.show()

        # Print values for analysis
        print("\nOptimized radius values (uniform gravity):")
        print(f"Min: {np.min(optimized_r):.0f} m, Max: {np.max(optimized_r):.0f} m")
        print(f"First few values: {optimized_r[:5]}")
        print(f"Middle values: {optimized_r[len(optimized_r)//2-2:len(optimized_r)//2+3]}")
        print(f"Last few values: {optimized_r[-5:]}")
        
        # Check symmetry
        mid_point = len(optimized_r) // 2
        left_half = optimized_r[:mid_point]
        right_half = optimized_r[mid_point:][::-1]  # Reverse the right half
        
        # If lengths are different (odd number of points), truncate
        min_length = min(len(left_half), len(right_half))
        left_half = left_half[:min_length]
        right_half = right_half[:min_length]
        
        symmetry_diff = np.abs(left_half - right_half)
        print(f"\nSymmetry check - Average difference: {np.mean(symmetry_diff):.2f} m")
        print(f"Max symmetry difference: {np.max(symmetry_diff):.2f} m")

        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity


def plot_prem_model():
    """Plot the brachistochrone curve for PREM gravity model"""
    print("Generating PREM model plot...")
    
    # Setup required variables
    num_points = 50
    r_initial = brachistochrone_initial_guess(R, num_points)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Set to PREM model
        globals()['velocity_improved'] = velocity_prem
        
        # Plot initial guess and Earth
        ax = setup_earth_plot('Brachistochrone Path with PREM Gravity Model')
        plt.plot(r_initial * np.cos(alpha_values), 
                r_initial * np.sin(alpha_values), 
                'k--', linewidth=1.5, label='Initial Guess')
        
        # Optimize path
        print("Optimizing PREM model path...")
        optimized_r = optimize_path_normalized(r_initial, alpha_values)
        travel_time = time_integral(optimized_r, alpha_values) / 60
        print(f"Travel time with PREM model: {travel_time:.2f} minutes")
        
        # Plot the optimized path
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                'b-', linewidth=3, label=f'PREM Model (Travel time: {travel_time:.2f} min)')
        
        plt.plot(optimized_r * np.cos(alpha_values), 
                optimized_r * np.sin(alpha_values), 
                 'm--', linewidth=1.5, label='Theoretical Brachistochrone')

        plt.legend(loc='upper right')
        plt.savefig('prem_model.png')
        plt.show()

        # Print values for analysis
        print("\nOptimized radius values (uniform gravity):")
        print(f"Min: {np.min(optimized_r):.0f} m, Max: {np.max(optimized_r):.0f} m")
        print(f"First few values: {optimized_r[:5]}")
        print(f"Middle values: {optimized_r[len(optimized_r)//2-2:len(optimized_r)//2+3]}")
        print(f"Last few values: {optimized_r[-5:]}")
        
        # Check symmetry
        mid_point = len(optimized_r) // 2
        left_half = optimized_r[:mid_point]
        right_half = optimized_r[mid_point:][::-1]  # Reverse the right half
        
        # If lengths are different (odd number of points), truncate
        min_length = min(len(left_half), len(right_half))
        left_half = left_half[:min_length]
        right_half = right_half[:min_length]
        
        symmetry_diff = np.abs(left_half - right_half)
        print(f"\nSymmetry check - Average difference: {np.mean(symmetry_diff):.2f} m")
        print(f"Max symmetry difference: {np.max(symmetry_diff):.2f} m")

        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity


def plot_all_models_comparison():
    """Plot all three gravity models on the same chart for easy comparison"""
    print("Generating comparison plot of all gravity models...")
    
    # Setup required variables
    num_points = 50
    r_initial = brachistochrone_initial_guess(R, num_points)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Setup Earth plot
        ax = setup_earth_plot('Comparison of Brachistochrone Paths for Different Gravity Models')
        
        # Uniform Density Model
        globals()['velocity_improved'] = velocity_uniform_density
        optimized_r_uniform_density = optimize_path_normalized(r_initial, alpha_values)
        travel_time_uniform_density = time_integral(optimized_r_uniform_density, alpha_values) / 60
        plt.plot(optimized_r_uniform_density * np.cos(alpha_values), 
                optimized_r_uniform_density * np.sin(alpha_values), 
                'g-', linewidth=2, label=f'Uniform Density ({travel_time_uniform_density:.2f} min)')
        
        # PREM Model
        globals()['velocity_improved'] = velocity_prem
        optimized_r_prem = optimize_path_normalized(r_initial, alpha_values)
        travel_time_prem = time_integral(optimized_r_prem, alpha_values) / 60
        plt.plot(optimized_r_prem * np.cos(alpha_values), 
                optimized_r_prem * np.sin(alpha_values), 
                'b-', linewidth=2, label=f'PREM Model ({travel_time_prem:.2f} min)')
        
        plt.plot(optimized_r_prem * np.cos(alpha_values), 
                optimized_r_prem * np.sin(alpha_values), 
                 'm--', linewidth=1.5, label='Theoretical Brachistochrone')
        
        plt.legend(loc='upper right')
        plt.savefig('all_models_comparison.png', dpi=300)
        plt.show()
        
        # Print comparison summary
        print("\n===== MODEL COMPARISON SUMMARY =====")
        print(f"Uniform Density: Travel time = {travel_time_uniform_density:.2f} minutes")
        print(f"                 Min depth = {R - np.min(optimized_r_uniform_density):.0f} m")
        print(f"PREM Model:      Travel time = {travel_time_prem:.2f} minutes")
        print(f"                 Min depth = {R - np.min(optimized_r_prem)::.0f} m")
        
        # Save optimized paths data
        np.savez('optimized_paths.npz', 
                 alpha_values=alpha_values,
                 uniform_density_r=optimized_r_uniform_density,
                 prem_r=optimized_r_prem)
        
        return optimized_r_uniform_density, optimized_r_prem, alpha_values
        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

def calculate_centripetal_acceleration(r_values, alpha_values):
    """Calculate centripetal acceleration at each point along the path"""
    # Get velocities at each point
    v_values = velocity_improved(r_values)
    
    # Calculate radius of curvature (approximation)
    # For a particle moving along r(α), the centripetal acceleration is approximately v²/ρ
    # where ρ is the radius of curvature
    
    # First, compute dr/dα (using finite differences)
    dr_dalpha = np.zeros_like(r_values)
    dr_dalpha[1:-1] = (r_values[2:] - r_values[:-2]) / (alpha_values[2:] - alpha_values[:-2])
    dr_dalpha[0] = (r_values[1] - r_values[0]) / (alpha_values[1] - alpha_values[0])
    dr_dalpha[-1] = (r_values[-1] - r_values[-2]) / (alpha_values[-1] - alpha_values[-2])
    
    # Second derivative (for curvature calculation)
    d2r_dalpha2 = np.zeros_like(r_values)
    d2r_dalpha2[1:-1] = (r_values[2:] - 2*r_values[1:-1] + r_values[:-2]) / ((alpha_values[2:] - alpha_values[:-2])/2)**2
    d2r_dalpha2[0] = d2r_dalpha2[1]
    d2r_dalpha2[-1] = d2r_dalpha2[-2]
    
    # Radius of curvature formula for polar coordinates
    denominator = np.abs(r_values**2 + 2*(dr_dalpha)**2 - r_values*d2r_dalpha2)
    denominator = np.maximum(denominator, 1e-10)  # Prevent division by zero
    radius_of_curvature = (r_values**2 + dr_dalpha**2)**(1.5) / denominator
    
    # Calculate centripetal acceleration (a = v²/r)
    centripetal_acceleration = v_values**2 / radius_of_curvature
    
    return centripetal_acceleration

def plot_centripetal_acceleration():
    """Plot centripetal acceleration for all three gravity models"""
    print("Analyzing centripetal acceleration for different gravity models...")
    
    # Load the optimized paths if available, otherwise run optimization
    try:
        data = np.load('optimized_paths.npz')
        alpha_values = data['alpha_values']
        uniform_gravity_r = data['uniform_gravity_r']
        uniform_density_r = data['uniform_density_r']
        prem_r = data['prem_r']
    except:
        print("No saved paths found. Running optimization...")
        uniform_density_r, prem_r, alpha_values = plot_all_models_comparison()
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Setup plot
        plt.figure(figsize=(10, 6))
        
        # Calculate acceleration for uniform density
        globals()['velocity_improved'] = velocity_uniform_density
        acc_uniform_density = calculate_centripetal_acceleration(uniform_density_r, alpha_values)
        
        # Calculate acceleration for PREM model
        globals()['velocity_improved'] = velocity_prem
        acc_prem = calculate_centripetal_acceleration(prem_r, alpha_values)
        
        # Convert alpha values to degrees for more intuitive x-axis
        alpha_degrees = alpha_values * 180 / np.pi
        
        # Plot accelerations
        plt.plot(alpha_degrees, acc_uniform_density, 'g-', label='Uniform Density')
        plt.plot(alpha_degrees, acc_prem, 'b-', label='PREM Model')
        
        # Add horizontal line for maximum human tolerance (9g ≈ 88.2 m/s²)
        max_human_g = 9  # 9g is typically considered maximum for trained pilots
        max_acc_human = max_human_g * 9.81  # Convert to m/s²
        plt.axhline(y=max_acc_human, color='k', linestyle='--', 
                   label=f'Max Human Tolerance ({max_human_g}g)')
        
        # Find points where acceleration exceeds human tolerance
        exceeds_uniform_density = acc_uniform_density > max_acc_human
        exceeds_prem = acc_prem > max_acc_human
        
        # Print warnings if any points exceed the tolerance
        if np.any(exceeds_uniform_density):
            print(f"WARNING: Uniform density model exceeds human tolerance at {np.sum(exceeds_uniform_density)} points")
            print(f"Max acceleration: {np.max(acc_uniform_density)/9.81:.1f}g")
        
        if np.any(exceeds_prem):
            print(f"WARNING: PREM model exceeds human tolerance at {np.sum(exceeds_prem)} points")
            print(f"Max acceleration: {np.max(acc_prem)/9.81:.1f}g")
        
        plt.xlabel('Angle Along Path (degrees)')
        plt.ylabel('Centripetal Acceleration (m/s²)')
        plt.title('Centripetal Acceleration Along Brachistochrone Paths')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('centripetal_acceleration.png', dpi=300)
        plt.show()
        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

def calculate_full_acceleration(r_values, alpha_values):
    """Calculate full acceleration (centripetal + tangential) at each point along the path"""
    # Get velocities at each point
    v_values = velocity_improved(r_values)
    
    # Calculate radius of curvature and centripetal acceleration (vectorized)
    # First, compute dr/dα (using finite differences)
    dr_dalpha = np.zeros_like(r_values)
    dr_dalpha[1:-1] = (r_values[2:] - r_values[:-2]) / (alpha_values[2:] - alpha_values[:-2])
    dr_dalpha[0] = (r_values[1] - r_values[0]) / (alpha_values[1] - alpha_values[0])
    dr_dalpha[-1] = (r_values[-1] - r_values[-2]) / (alpha_values[-1] - alpha_values[-2])
    
    # Second derivative (for curvature calculation)
    d2r_dalpha2 = np.zeros_like(r_values)
    d2r_dalpha2[1:-1] = (r_values[2:] - 2*r_values[1:-1] + r_values[:-2]) / ((alpha_values[2:] - alpha_values[:-2])/2)**2
    d2r_dalpha2[0] = d2r_dalpha2[1]
    d2r_dalpha2[-1] = d2r_dalpha2[-2]
    
    # Unit vectors in polar coordinates 
    # e_r = (cos(α), sin(α))
    # e_θ = (-sin(α), cos(α))
    e_r_x = np.cos(alpha_values)
    e_r_y = np.sin(alpha_values)
    e_theta_x = -np.sin(alpha_values)
    e_theta_y = np.cos(alpha_values)
    
    # Radius of curvature formula for polar coordinates
    denominator = np.abs(r_values**2 + 2*(dr_dalpha)**2 - r_values*d2r_dalpha2)
    denominator = np.maximum(denominator, 1e-10)  # Prevent division by zero
    radius_of_curvature = (r_values**2 + dr_dalpha**2)**(1.5) / denominator
    
    # Calculate centripetal acceleration vector (a_c = v²/r)
    # Direction is towards center of curvature
    centripetal_mag = v_values**2 / radius_of_curvature
    
    # Get tangential acceleration (due to gravity)
    if globals()['velocity_improved'] == velocity_uniform_gravity:
        # For uniform gravity, acceleration is constant magnitude but direction changes
        g_mag = G * M / (R**2)  # Constant gravity magnitude
        # Project onto tangent direction
        g_tangential = g_mag * np.abs(np.sin(alpha_values))
        
    elif globals()['velocity_improved'] == velocity_uniform_density:
        # For uniform density model, gravity increases with depth
        g_mag = np.array([uniform_density_function(r) for r in r_values])
        # Project onto tangent direction
        g_tangential = g_mag * np.abs(np.sin(alpha_values))
        
    else:  # PREM model
        # For PREM model, use the PREM gravity function
        g_mag = np.array([prem_gravity_function(r) for r in r_values])
        # Project onto tangent direction
        g_tangential = g_mag * np.abs(np.sin(alpha_values))
    
    # Total acceleration magnitude (vector sum of centripetal and tangential)
    total_acceleration = np.sqrt(centripetal_mag**2 + g_tangential**2)
    
    return total_acceleration, centripetal_mag, g_tangential

def plot_full_acceleration():
    """Plot full acceleration for all three gravity models with human comfort levels"""
    print("Analyzing full acceleration for different gravity models...")
    
    # Load the optimized paths if available, otherwise run optimization
    try:
        data = np.load('optimized_paths.npz')
        alpha_values = data['alpha_values']
        uniform_density_r = data['uniform_density_r']
        prem_r = data['prem_r']
    except:
        print("No saved paths found. Running optimization...")
        uniform_density_r, prem_r, alpha_values = plot_all_models_comparison()
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        # Setup plot
        plt.figure(figsize=(12, 8))

        # Calculate acceleration for uniform density
        globals()['velocity_improved'] = velocity_uniform_density
        acc_uniform_density, cent_uniform_density, tang_uniform_density = calculate_full_acceleration(uniform_density_r, alpha_values)
        
        # Calculate acceleration for PREM model
        globals()['velocity_improved'] = velocity_prem
        acc_prem, cent_prem, tang_prem = calculate_full_acceleration(prem_r, alpha_values)
        
        # Convert alpha values to degrees for more intuitive x-axis
        alpha_degrees = alpha_values * 180 / np.pi
        
        # First, add the comfort level lines with labels directly on the lines
        comfort_levels = [
            (1, 'Comfortable', 'green', '--'),
            (2, 'Uncomfortable', 'yellow', '--'),
            (5, 'Very Uncomfortable', 'orange', '--'),
            (9, 'Maximum for Trained Pilots', 'red', '--')
        ]
        
        # Plot acceleration curves first so comfort lines appear on top
        plt.plot(alpha_degrees, acc_uniform_density, 'g-', linewidth=2, label='Uniform Density')
        plt.plot(alpha_degrees, acc_prem, 'b-', linewidth=2, label='PREM Model')
        
        # Then plot comfort level lines
        for g_level, label, color, linestyle in comfort_levels:
            acc_level = g_level * 9.81  # Convert to m/s²
            line = plt.axhline(y=acc_level, color=color, linestyle=linestyle, alpha=0.7)
            
            # Add text near the start of the graph instead of the end
            text_x = alpha_degrees[-1] * 0.1  # Position at 10% along x-axis
            text_y = acc_level + 2  # Position slightly above the line
            plt.text(text_x, text_y, f'{label} ({g_level}g)', 
                   color=color, fontweight='bold', va='bottom', ha='left',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        
        # Set y-axis to a reasonable range
        plt.ylim(0, 100)
        
        # Set x-axis range to match data
        plt.xlim(0, alpha_degrees[-1])
        
        # Find points where acceleration exceeds human tolerance
        max_human_g = 9
        max_acc_human = max_human_g * 9.81
        
        for model_name, acceleration in [
            ('Uniform Density', acc_uniform_density),
            ('PREM Model', acc_prem)
        ]:
            exceeds = acceleration > max_acc_human
            if np.any(exceeds):
                print(f"WARNING: {model_name} exceeds human tolerance at {np.sum(exceeds)} points")
                print(f"Max acceleration: {np.max(acceleration)/9.81:.1f}g at angle: {alpha_degrees[np.argmax(acceleration)]:.1f}°")
                print(f"Average acceleration: {np.mean(acceleration)/9.81:.1f}g")
        
        plt.xlabel('Angle Along Path (degrees)')
        plt.ylabel('Total Acceleration (m/s²)')
        plt.title('Total Acceleration Along Brachistochrone Paths')
        plt.grid(True)
        
        # Only include model curves in the legend, not comfort levels
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('total_acceleration.png', dpi=300)
        plt.show()
        
        # Also update the component plot labels similarly
        plt.figure(figsize=(12, 8))
        # ... existing code ...
        
        # Add comfort lines to this plot too with consistent positioning
        for g_level, label, color, linestyle in comfort_levels:
            acc_level = g_level * 9.81  # Convert to m/s²
            line = plt.axhline(y=acc_level, color=color, linestyle=linestyle, alpha=0.7)
            
            # Add text near the start of the graph 
            text_x = alpha_degrees[-1] * 0.1  # Position at 10% along x-axis
            text_y = acc_level + 2  # Position slightly above the line
            plt.text(text_x, text_y, f'{label} ({g_level}g)', 
                   color=color, fontweight='bold', va='bottom', ha='left',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        
        # ... rest of function ...
        
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

# Call this new function after comparing paths
if __name__ == "__main__":
    # Individual models
    #plot_uniform_gravity_model()
    #plot_uniform_density_model()
    #plot_prem_model()
    
    # Comparison plot
    uniform_density_r, prem_r, alpha_values = plot_all_models_comparison()
    
    # Calculate and plot centripetal acceleration
    #plot_centripetal_acceleration()
    
    # Calculate and plot full acceleration
    #plot_full_acceleration()
    
    # Optional: Calculate and display the path differences
    #max_diff_gravity_density = np.max(np.abs(uniform_gravity_r - uniform_density_r))
    #max_diff_gravity_prem = np.max(np.abs(uniform_gravity_r - prem_r))
    #max_diff_density_prem = np.max(np.abs(uniform_density_r - prem_r))
    
    #print("\n===== PATH DIFFERENCES =====")
    #print(f"Max difference between Uniform Gravity and Uniform Density: {max_diff_gravity_density:.0f} m")
    #print(f"Max difference between Uniform Gravity and PREM: {max_diff_gravity_prem:.0f} m")
    #print(f"Max difference between Uniform Density and PREM: {max_diff_density_prem:.0f} m")