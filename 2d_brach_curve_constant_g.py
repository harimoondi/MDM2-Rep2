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
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping

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

def calculate_ep(r: list): # Used to validate np.trapz is integrating correctly
    r = np.array(r)
    return np.where(r == 0, 0, G * M * (r - R) / R ** 2)

def velocity(r_values):
    min_depth_index = np.argmin(r_values)
    potential_energy = np.zeros_like(r_values)

    for i in range(1, min_depth_index + 1):
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = uniform_density_function(segment_r)  # Use PREM gravity function instead
        potential_energy[i] = np.trapz(segment_g, segment_r)

    for i in range(min_depth_index + 1, len(r_values)):
        segment_r = np.linspace(r_values[0], r_values[i], 100)
        segment_g = uniform_density_function(segment_r)  # Use PREM gravity function instead
        potential_energy[i] = np.trapz(segment_g, segment_r)

    kinetic_energy = 0 - potential_energy
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

def velocity_normalized(r_values):
    """Simplified velocity calculation for uniform gravity"""
    # Normalized depth (0 at surface, positive as you go deeper)
    depth = (r_earth - r_values) / r_earth
    
    # For uniform gravity field, velocity is proportional to sqrt(depth)
    # Using v = sqrt(2*g*h) but normalized
    g = 9.81  # m/s²
    velocity = np.sqrt(2 * g * depth * r_earth)
    
    return velocity

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

def smoothness_penalty(r_values, alpha_values, penalty_weight=1e-7):  # Reduced from 1e-5
    """Add a penalty for zigzagging paths"""
    # Calculate second derivative as a measure of smoothness
    second_derivatives = np.diff(np.diff(r_values))
    # Return the sum of squares of second derivatives as penalty
    return penalty_weight * np.sum(second_derivatives**2)

def objective_with_smoothing(params, alpha_values):
    r_values = params.copy()
    r_values[0] = r_values[-1] = r_earth
    
    # Calculate travel time
    total_time = time_integral(r_values, alpha_values)
    
    # Add smoothness penalty
    penalty = smoothness_penalty(r_values, alpha_values)
    
    return total_time + penalty

def objective(params, alpha_values):
    r_values = params
    r_values[0] = r_values[-1] = r_earth
    total_time = time_integral(r_values, alpha_values)
    return total_time

def optimize_path(r_init, alpha_values):
    n_points = len(r_init)
    bounds = [(0.1 * r_earth, r_earth) for _ in range(n_points)]
    bounds[0] = bounds[-1] = (r_earth, r_earth)
    
    # Create a minimizer function that respects bounds
    def bounded_objective(x):
        # Apply bounds
        x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        # Fix endpoints
        x[0] = x[-1] = r_earth
        return objective_with_smoothing(x, alpha_values)
    
    # Use basin hopping with local minimizer
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    result = basinhopping(
        bounded_objective, 
        r_init,
        niter=10,
        T=10.0,
        stepsize=0.5 * r_earth,
        minimizer_kwargs=minimizer_kwargs,
        disp=True
    )
    
    optimized_r_values = result.x
    optimized_r_values[0] = optimized_r_values[-1] = r_earth
    return optimized_r_values

def optimize_path_global(r_init, alpha_values):
    n_points = len(r_init)
    bounds = [(0.1 * r_earth, r_earth) for _ in range(n_points)]
    # Fix the boundary points
    bounds[0] = (r_earth, r_earth)
    bounds[-1] = (r_earth, r_earth)
    
    def obj_wrapper(params):
        params_copy = params.copy()
        params_copy[0] = params_copy[-1] = r_earth
        return objective(params_copy, alpha_values)
    
    result = differential_evolution(
        obj_wrapper,
        bounds,
        popsize=15,
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        maxiter=100,
        disp=True
    )
    
    optimized_r_values = result.x
    optimized_r_values[0] = optimized_r_values[-1] = r_earth
    return optimized_r_values

def optimize_path_robust(r_init, alpha_values):
    """More robust optimization approach"""
    n_points = len(r_init)
    bounds = [(0.1 * r_earth, r_earth) for _ in range(n_points)]
    bounds[0] = bounds[-1] = (r_earth, r_earth)
    
    # Step 1: Use direct minimization with SLSQP first
    def obj_func(x):
        x_copy = x.copy()
        x_copy[0] = x_copy[-1] = r_earth
        return time_integral(x_copy, alpha_values)
    
    # Add constraint that r values must decrease then increase (to prevent zigzags)
    def constraint(x):
        # Find minimum point
        min_idx = np.argmin(x)
        # Before min: should be decreasing
        before_constraint = np.diff(x[:min_idx+1]) <= 0
        # After min: should be increasing
        after_constraint = np.diff(x[min_idx:]) >= 0
        return np.all(np.concatenate([before_constraint, after_constraint]))
    
    # First optimization with strong shape constraints
    result = minimize(
        obj_func,
        r_init,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-8, 'disp': True}
    )
    
    # Try a second optimization with smaller steps and the improved result
    minimizer_kwargs = {
        "method": "L-BFGS-B", 
        "bounds": bounds,
        "options": {
            'maxiter': 1000,
            'ftol': 1e-8
        }
    }
    
    result = basinhopping(
        obj_func, 
        result.x,  # Use result from first optimization
        niter=20,
        T=1.0,     # Lower temperature
        stepsize=0.05 * r_earth,  # Much smaller stepsize
        minimizer_kwargs=minimizer_kwargs,
        disp=True
    )
    
    optimized_r_values = result.x
    optimized_r_values[0] = optimized_r_values[-1] = r_earth
    return optimized_r_values

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


def close_initial_guess(R):
    # Define fixed points
    x, y, magnitude = theoretical_brachistochrone(R)
    fixed_indices = [0, 25, 50, 100, 150, 175, 199]
    # fixed_values = [R, magnitude[25], magnitude[50], magnitude[100], magnitude[150], magnitude[175], R]
    fixed_values = [R,  4500000, 3400000, 3500000, 3400000, 4500000,  R]

    # Create cubic spline interpolation
    spline = CubicSpline(fixed_indices, fixed_values, bc_type='clamped')

    # Generate smooth curve
    r_initial = spline(np.arange(200))

    # Add small noise for variability
    r_initial += np.random.uniform(-1e-2, 1e-2, size=r_initial.shape)

    return r_initial

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

def zigzag_initial_guess(R, num_points=200):
    """Create a zigzag initial guess that's deliberately far from optimal"""
    theta = np.linspace(0, np.pi/2, num_points)
    
    # Create a zigzag pattern with multiple oscillations
    oscillations = 3
    amplitude = 0.4 * R  # 40% of Earth radius
    
    r_initial = R - amplitude * np.sin(oscillations * theta)
    
    # Ensure boundary conditions
    r_initial[0] = r_initial[-1] = R
    
    return r_initial

def cycloid_initial_guess(num_points=200):
    """Generate a cycloid-like initial guess, normalized between 0 and 1"""
    t = np.linspace(0, np.pi, num_points)
    
    # Simple cycloid-like curve: y = sin(t/2)²
    depth = 0.3 * np.sin(t/2)**2  # Max depth of 30%
    
    # Convert to radius (1.0 = Earth surface)
    r_normalized = 1.0 - depth
    
    # Convert back to meters
    r_initial = r_normalized * r_earth
    
    # Ensure endpoints
    r_initial[0] = r_initial[-1] = r_earth
    
    return r_initial

def surface_initial_guess(num_points=200):
    """Create an initial guess that follows Earth's surface (great circle)
    
    This path stays at r=R for all points (Earth's surface), representing
    travel along a great circle route between the endpoints.
    """
    # Create a path that stays at Earth's radius
    r_initial = np.ones(num_points) * r_earth
    
    # Optional: Add tiny perturbation to help optimizer
    # r_initial[1:-1] -= np.random.uniform(1, 100, size=num_points-2)
    
    return r_initial

def chebyshev_nodes(a, b, n):
    """Generate Chebyshev nodes in the interval [a, b]
    
    This creates nodes that are more densely distributed near the edges
    of the interval, which is useful for better approximation at boundaries.
    """
    k = np.arange(n)
    nodes = (a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) * np.pi / (2 * n))
    # Return sorted nodes (Chebyshev nodes are naturally ordered from b to a)
    return np.sort(nodes)

# Create a separate function for the comparison to make it more robust
def plot_gravity_model_comparison():
    print("Starting gravity model comparison...")
    
    # Initial guess - simple parabolic dip for speed
    num_points = 100  # Reduced from 200 for faster optimization
    r_initial = straight_line_initial_guess(R, num_points)
    
    # Use fewer angle steps for faster calculation
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = np.linspace(0, central_angle, num_points)  # Use linear spacing for simplicity
    
    # Try to optimize with each gravity model, with error handling
    try:
        print("Optimizing uniform gravity model...")
        optimized_r_uniform_gravity = optimize_with_gravity_model(r_initial, alpha_values, velocity_uniform_gravity)
        
        print("Optimizing uniform density model...")
        optimized_r_uniform_density = optimize_with_gravity_model(r_initial, alpha_values, velocity_uniform_density)
        
        print("Optimizing PREM model...")
        optimized_r_prem = optimize_with_gravity_model(r_initial, alpha_values, velocity_prem)
        
        # Create comparative plot
        plt.figure(figsize=(14, 10))
        
        # Plot Earth's surface
        theta = np.linspace(0, 2 * np.pi, 1000)
        x_circle = r_earth * np.cos(theta)
        y_circle = r_earth * np.sin(theta)
        plt.plot(x_circle, y_circle, 'k-', linewidth=1, label='Earth Surface')
        
        # Plot the optimized paths
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
        x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
        plt.plot(x_theoretical, y_theoretical, 'm--', linewidth=1, label='Theoretical Brachistochrone')
        
        # Mark the start and end points
        plt.plot(0, r_earth, 'ko', markersize=8)
        plt.plot(r_earth, 0, 'ko', markersize=8)
        
        # Zoom in to show the differences better
        zoom_factor = 0.2  # Adjust this to control zoom level
        plt.xlim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
        plt.ylim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
        
        plt.title('Comparison of Brachistochrone Paths for Different Gravity Models')
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.savefig('gravity_model_comparison.png')  # Save the figure in case the display fails
        plt.show()
        
        # Print travel times for each model
        original_velocity = globals()['velocity_improved']
        
        globals()['velocity_improved'] = velocity_uniform_gravity
        time_uniform_gravity = time_integral(optimized_r_uniform_gravity, alpha_values)/60
        
        globals()['velocity_improved'] = velocity_uniform_density
        time_uniform_density = time_integral(optimized_r_uniform_density, alpha_values)/60
        
        globals()['velocity_improved'] = velocity_prem
        time_prem = time_integral(optimized_r_prem, alpha_values)/60
        
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity
        
        print(f"Travel times (minutes):")
        print(f"Uniform Gravity: {time_uniform_gravity:.2f}")
        print(f"Uniform Density: {time_uniform_density:.2f}")
        print(f"PREM Model: {time_prem:.2f}")
        
    except Exception as e:
        print(f"Error during gravity model comparison: {e}")

def plot_gravity_model_comparison_fast():
    """Faster version of gravity model comparison with performance optimizations"""
    print("Starting optimized gravity model comparison...")
    
    # Use fewer points for faster calculation
    num_points = 50  # Reduced from 100 for much faster optimization
    r_initial = straight_line_initial_guess(R, num_points)
    
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = np.linspace(0, central_angle, num_points)
    
    # Store original velocity function
    original_velocity = globals()['velocity_improved']
    
    # Optimization settings - use looser tolerances for speed
    def optimize_quick(r_init, alpha_values, velocity_func):
        # Temporarily replace velocity function
        globals()['velocity_improved'] = velocity_func
        
        # Use direct minimization with fewer iterations
        n_points = len(r_init)
        bounds = [(0.1, 1.0) for _ in range(n_points)]
        bounds[0] = bounds[-1] = (1.0, 1.0)
        
        # Normalized values for better stability
        r_init_norm = r_init / r_earth
        
        def normalized_objective(x_norm):
            x_real = x_norm * r_earth
            x_real[0] = x_real[-1] = r_earth
            return time_integral(x_real, alpha_values)
        
        result = minimize(
            normalized_objective,
            r_init_norm,
            method='SLSQP',
            bounds=bounds,
            options={
                'ftol': 1e-6,  # Looser tolerance
                'maxiter': 200,  # Fewer iterations
                'disp': True
            }
        )
        
        optimized_r = result.x * r_earth
        optimized_r[0] = optimized_r[-1] = r_earth
        return optimized_r
    
    # Run optimizations with simplified settings
    try:
        print("Optimizing uniform gravity model...")
        optimized_r_uniform_gravity = optimize_quick(r_initial, alpha_values, velocity_uniform_gravity)
        
        print("Optimizing uniform density model...")
        optimized_r_uniform_density = optimize_quick(r_initial, alpha_values, velocity_uniform_density)
        
        print("Optimizing PREM model...")
        optimized_r_prem = optimize_quick(r_initial, alpha_values, velocity_prem)
        
        # Plotting code (same as before)
        plt.figure(figsize=(14, 10))
        
        # Plot Earth's surface
        theta = np.linspace(0, 2 * np.pi, 1000)
        x_circle = r_earth * np.cos(theta)
        y_circle = r_earth * np.sin(theta)
        plt.plot(x_circle, y_circle, 'k-', linewidth=1, label='Earth Surface')
        
        # Plot optimized paths
        plt.plot(optimized_r_uniform_gravity * np.cos(alpha_values), 
                optimized_r_uniform_gravity * np.sin(alpha_values), 
                'r-', linewidth=2, label='Uniform Gravity')
                
        plt.plot(optimized_r_uniform_density * np.cos(alpha_values), 
                optimized_r_uniform_density * np.sin(alpha_values), 
                'g-', linewidth=2, label='Uniform Density')
                
        plt.plot(optimized_r_prem * np.cos(alpha_values), 
                optimized_r_prem * np.sin(alpha_values), 
                'b-', linewidth=2, label='PREM Model')
        
        # Add theoretical curve
        x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
        plt.plot(x_theoretical, y_theoretical, 'm--', linewidth=1, label='Theoretical Brachistochrone')
        
        # Mark endpoints
        plt.plot(0, r_earth, 'ko', markersize=8)
        plt.plot(r_earth, 0, 'ko', markersize=8)
        
        # Zoom in
        zoom_factor = 0.2
        plt.xlim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
        plt.ylim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
        
        plt.title('Comparison of Brachistochrone Paths for Different Gravity Models')
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.savefig('gravity_model_comparison.png')
        plt.show()
        
        # Calculate and display travel times
        globals()['velocity_improved'] = velocity_uniform_gravity
        time_uniform_gravity = time_integral(optimized_r_uniform_gravity, alpha_values)/60
        
        globals()['velocity_improved'] = velocity_uniform_density
        time_uniform_density = time_integral(optimized_r_uniform_density, alpha_values)/60
        
        globals()['velocity_improved'] = velocity_prem
        time_prem = time_integral(optimized_r_prem, alpha_values)/60
        
        print(f"Travel times (minutes):")
        print(f"Uniform Gravity: {time_uniform_gravity:.2f}")
        print(f"Uniform Density: {time_uniform_density:.2f}")
        print(f"PREM Model: {time_prem:.2f}")
        
    except Exception as e:
        print(f"Error during gravity model comparison: {e}")
    finally:
        # Restore original velocity function
        globals()['velocity_improved'] = original_velocity

def plot_simple_gravity_comparison():
    print("Starting gravity model comparison...")
    
    # Use the same point count as the initial optimization
    num_points = 200
    r_initial = straight_line_initial_guess(R, num_points)
    
    # Use the same alpha distribution as the initial optimization
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, num_points)
    
    # Save original velocity function
    original_velocity = globals()['velocity_improved']
    
    try:
        print("Optimizing uniform gravity model...")
        # Use the same optimization method as the initial plot
        globals()['velocity_improved'] = velocity_uniform_gravity
        optimized_r_uniform_gravity = optimize_path_normalized(r_initial, alpha_values)
        time_uniform_gravity = time_integral(optimized_r_uniform_gravity, alpha_values)/60
        
        print("Optimizing uniform density model...")
        globals()['velocity_improved'] = velocity_uniform_density
        optimized_r_uniform_density = optimize_path_normalized(r_initial, alpha_values)
        time_uniform_density = time_integral(optimized_r_uniform_density, alpha_values)/60
        
        print("Optimizing PREM model...")
        globals()['velocity_improved'] = velocity_prem
        optimized_r_prem = optimize_path_normalized(r_initial, alpha_values)
        time_prem = time_integral(optimized_r_prem, alpha_values)/60
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Plot Earth's surface
        theta = np.linspace(0, 2 * np.pi, 1000)
        x_circle = r_earth * np.cos(theta)
        y_circle = r_earth * np.sin(theta)
        plt.plot(x_circle, y_circle, 'k-', linewidth=1, label='Earth Surface')
        
        # Plot the optimized paths
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
        plt.savefig('gravity_model_comparison.png')
        plt.show()
        
        # Print travel times
        print(f"Travel times (minutes):")
        print(f"Uniform Gravity: {time_uniform_gravity:.2f}")
        print(f"Uniform Density: {time_uniform_density:.2f}")
        print(f"PREM Model: {time_prem:.2f}")
        
    except Exception as e:
        print(f"Error during gravity model comparison: {e}")
    finally:
        # Always restore original velocity function
        globals()['velocity_improved'] = original_velocity

def run_gravity_model_comparison():
    """Run full optimization and comparison of different gravity models.
    This function performs the full optimization for three different gravity models
    and plots the results. It's separated into a function to avoid running these
    time-consuming optimizations automatically."""
    
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

    # Initial guess
    r_initial = straight_line_initial_guess(R)
    print('R_initial', r_initial)

    # Generate alpha values
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)

    # With Chebyshev nodes for better edge resolution
    alpha_values = chebyshev_nodes(0, central_angle, len(r_initial))

    # First plot - single optimization
    plt.figure(figsize=(12, 12))

    # Optimize the path
    optimized_r = optimize_path_normalized(r_initial, alpha_values)
    travel_time = time_integral(optimized_r, alpha_values)
    max_depth = r_earth - np.min(optimized_r)

    # Plot the optimized tunnel path
    plt.plot(optimized_r * np.cos(alpha_values), optimized_r * np.sin(alpha_values), 'r-', linewidth=2, label='Optimized Tunnel')

    # Calculate MSE between optimized curve and theoretical brachistochrone
    x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
    theoretical_r = np.sqrt(x_theoretical ** 2 + y_theoretical ** 2)
    interp_func = interp1d(np.linspace(0, 1, len(theoretical_r)), theoretical_r)
    resampled_theoretical_r = interp_func(np.linspace(0, 1, len(optimized_r)))
    mse = np.mean((optimized_r - resampled_theoretical_r) ** 2)
    print(f'Mean Squared Error (MSE) between optimized and theoretical curve: {mse:.4f}')

    # Print the results
    print("Travel Time (mins):", travel_time / 60)
    print("Max Depth (m):", max_depth)

    # Plot the earths surface
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = r_earth * np.cos(theta)
    y_circle = r_earth * np.sin(theta)
    plt.plot(x_circle, y_circle, 'k-', linewidth=1)

    # Plot the theoretical brachistochrone
    plt.plot(x_theoretical, y_theoretical, 'm--', linewidth=2, label='Theoretical Brachistochrone')

    # Plot initial guess
    plt.plot(r_initial * np.cos(alpha_values), r_initial * np.sin(alpha_values), 'b--', linewidth=1, label='Initial Guess')

    # Mark the start and end points
    plt.plot(0, r_earth, 'ro', label='Start')
    plt.plot(r_earth, 0, 'go', label='End')

    plt.title('Model Validation by Comparing Optimized Tunnel in a '
            'Uniform Gravity Field to Theoretical Brachistochrone')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Main execution code for comparison
    # Initial guess and alpha values setup remain the same
    r_initial = straight_line_initial_guess(R)
    central_angle = central_angle_calculation(theta_init, phi_init, theta_end, phi_end)
    alpha_values = chebyshev_nodes(0, central_angle, len(r_initial))

    # Optimize for each gravity model
    print("Running optimizations for three gravity models (this may take a while)...")
    optimized_r_uniform_gravity = optimize_with_gravity_model(r_initial, alpha_values, velocity_uniform_gravity)
    optimized_r_uniform_density = optimize_with_gravity_model(r_initial, alpha_values, velocity_uniform_density)
    optimized_r_prem = optimize_with_gravity_model(r_initial, alpha_values, velocity_prem)

    # Create comparative plot
    plt.figure(figsize=(14, 10))

    # Plot Earth's surface
    theta = np.linspace(0, 2 * np.pi, 1000)
    x_circle = r_earth * np.cos(theta)
    y_circle = r_earth * np.sin(theta)
    plt.plot(x_circle, y_circle, 'k-', linewidth=1, label='Earth Surface')

    # Plot the optimized paths
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
    x_theoretical, y_theoretical, _ = theoretical_brachistochrone(r_earth)
    plt.plot(x_theoretical, y_theoretical, 'm--', linewidth=1, label='Theoretical Brachistochrone')

    # Mark the start and end points
    plt.plot(0, r_earth, 'ko', markersize=8, label='Start')
    plt.plot(r_earth, 0, 'ko', markersize=8, label='End')

    # Zoom in to show the differences better
    zoom_factor = 0.2  # Adjust this to control zoom level
    plt.xlim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))
    plt.ylim(-r_earth * zoom_factor, r_earth * (1 + zoom_factor))

    plt.title('Comparison of Brachistochrone Paths for Different Gravity Models')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

    # Print travel times for each model
    original_velocity = globals()['velocity_improved']
    
    globals()['velocity_improved'] = velocity_uniform_gravity
    time_uniform_gravity = time_integral(optimized_r_uniform_gravity, alpha_values)/60
    
    globals()['velocity_improved'] = velocity_uniform_density
    time_uniform_density = time_integral(optimized_r_uniform_density, alpha_values)/60
    
    globals()['velocity_improved'] = velocity_prem
    time_prem = time_integral(optimized_r_prem, alpha_values)/60
    
    # Restore original velocity function
    globals()['velocity_improved'] = original_velocity

    print(f"Travel times (minutes):")
    print(f"Uniform Gravity: {time_uniform_gravity:.2f}")
    print(f"Uniform Density: {time_uniform_density:.2f}")
    print(f"PREM Model: {time_prem:.2f}")

# Note: This function is only defined here but not called automatically.
# To run the comparison, call run_gravity_model_comparison() from your code.

# Replace the if __name__ == "__main__" section with:
if __name__ == "__main__":
    # Your existing code for the first plot here...
    
    # Call the simplified comparison function
    plot_simple_gravity_comparison()