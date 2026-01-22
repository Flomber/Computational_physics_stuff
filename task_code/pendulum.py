"""Pendulum Simulation using Numerical Integration Methods

This module simulates the motion of a simple pendulum using numerical integration.
The pendulum is described by the coupled ODEs:
    φ̇ = p (angular velocity)
    ṗ = -sin(φ) (angular acceleration, with g/L = 1)
where φ is the angle and p is the angular momentum.

Supports multiple integration methods:
- Explicit: Euler, RK4
- Implicit: Trapezoidal
- Symplectic: Drift-Kick, Leapfrog, Verlet
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def solvePendel(theta_n, p_n, theta_star, p_star, dt, tol=1e-10, max_iter=50):
    """Solve for p_{n+1} in the trapezoidal rule using Newton-Raphson iteration.
    
    The trapezoidal rule for the pendulum leads to the implicit equation:
    p_{n+1} = p* + (dt/2) * (sin(theta_n) - sin(theta_{n+1}))
    where theta_{n+1} = theta* + (dt/2) * (p_{n+1} - p_n)
    
    This becomes a transcendental equation F(p_{n+1}) = 0 that we solve iteratively.
    
    Args:
        theta_n: Current angle φ_n
        p_n: Current angular momentum p_n
        theta_star: θ* = θ_n + dt * p_n
        p_star: p* = p_n - dt * sin(θ_n)
        dt: Time step size
        tol: Convergence tolerance (default: 1e-10)
        max_iter: Maximum number of iterations (default: 50)
        
    Returns:
        p_{n+1}: Next angular momentum value
    """
    # Initial guess: use the predictor p*
    p_next = p_star
    
    for iteration in range(max_iter):
        # Calculate theta_{n+1} based on current guess for p_{n+1}
        theta_next = theta_star + 0.5 * dt * (p_next - p_n)
        
        # Define F(p_{n+1}) = p_{n+1} - p* - (dt/2)*sin(theta_n) + (dt/2)*sin(theta_{n+1})
        # We want to find p_{n+1} such that F(p_{n+1}) = 0
        F = p_next - p_star - 0.5 * dt * np.sin(theta_n) + 0.5 * dt * np.sin(theta_next)
        
        # Derivative: F'(p_{n+1}) = 1 + (dt/2) * cos(theta_{n+1}) * (dt/2)
        F_prime = 1.0 + 0.25 * dt**2 * np.cos(theta_next)
        
        # Newton-Raphson update: p_{n+1}^(k+1) = p_{n+1}^(k) - F / F'
        p_next_new = p_next - F / F_prime
        
        # Check convergence
        if abs(p_next_new - p_next) < tol:
            return p_next_new
        
        p_next = p_next_new
    
    # If we reach here, we didn't converge
    print(f"Warning: Newton-Raphson did not converge after {max_iter} iterations")
    return p_next


def pendulum(num_steps=201, method='rk4', animation_time=20.0, initial_phi=0.0, initial_p=0.2):
    """Solve the pendulum equations of motion using numerical integration.
    
    Args:
        num_steps: Number of time steps for integration
        method: Integration method - 'euler', 'rk4', 'trapezoidal', 'drift-kick', 
                'leapfrog', or 'verlet' (default: 'rk4')
        animation_time: Total animation duration in seconds (default: 20.0)
        initial_phi: Initial angle φ(0) in radians (default: 0.0)
        initial_p: Initial angular momentum p(0) (default: 0.2)
    
    Computes the trajectory of a simple pendulum in phase space (φ, p)
    and visualizes the result as an animated plot showing:
    - Phase space trajectory with energy contours
    - Physical pendulum animation
    - Energy conservation over time
    """
    # Time discretization: num_steps time steps from t=0 to t=60
    t = np.linspace(0., 60., num_steps)
    
    # Preallocate arrays for angle φ(t) and angular momentum p(t)
    phi = np.zeros_like(t)
    p = np.zeros_like(t)
    
    # Set initial conditions
    phi[0] = initial_phi
    p[0] = initial_p
    
    def G(f, t):
        """Right-hand side of the pendulum equations of motion.
        
        Args:
            f: Array [φ, p] containing angle and angular momentum
            t: Time (not used in this autonomous system, but included for generality)
            
        Returns:
            Array [φ̇, ṗ] = [p, -sin(φ)] representing the time derivatives
        
        Note: For autonomous systems like the pendulum, the RHS doesn't depend on time,
        but we include t as a parameter to maintain compatibility with general ODE solvers.
        """
        phi_val, p_val = f
        return np.array([p_val, -np.sin(phi_val)])

    # Numerical integration
    dt = t[1] - t[0]  # Time step size
    
    print(f"Using {method.upper()} integration method with {num_steps} steps")
    
    for i in range(1, len(t)):
        # Get previous state values
        f_old = np.array([phi[i-1], p[i-1]])
        t_old = t[i-1]
        
        if method.lower() == 'euler':
            # Euler method: y_{n+1} = y_n + dt * f(y_n, t_n)
            f_new = f_old + dt * G(f_old, t_old)
            
        elif method.lower() == 'rk4':
            # Runge-Kutta 4th order method
            # k1 = f(y_n, t_n)
            k1 = G(f_old, t_old)
            
            # k2 = f(y_n + dt/2 * k1, t_n + dt/2)
            k2 = G(f_old + 0.5 * dt * k1, t_old + 0.5 * dt)
            
            # k3 = f(y_n + dt/2 * k2, t_n + dt/2)
            k3 = G(f_old + 0.5 * dt * k2, t_old + 0.5 * dt)
            
            # k4 = f(y_n + dt * k3, t_n + dt)
            k4 = G(f_old + dt * k3, t_old + dt)
            
            # y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            f_new = f_old + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        elif method.lower() == 'trapezoidal':
            # Trapezoidal rule (implicit method)
            # Uses predictor-corrector approach with Newton-Raphson iteration
            
            # Predictor step (explicit Euler)
            theta_star = f_old[0] + dt * f_old[1]
            p_star = f_old[1] - dt * np.sin(f_old[0])
            
            # Corrector step: solve implicit equation for p_{n+1}
            p_new = solvePendel(f_old[0], f_old[1], theta_star, p_star, dt)
            
            # Calculate theta_{n+1} from p_{n+1}
            theta_new = theta_star + 0.5 * dt * (p_new - f_old[1])
            
            f_new = np.array([theta_new, p_new])

        elif method.lower() == 'drift-kick':
            # Drift-Kick method (symplectic integrator)
            # Also known as symplectic Euler or semi-implicit Euler
            # Update position first (drift), then momentum (kick)
            phi_new = f_old[0] + dt * f_old[1]  # Drift: φ_{n+1} = φ_n + dt * p_n
            p_new = f_old[1] - dt * np.sin(phi_new)  # Kick: p_{n+1} = p_n - dt * sin(φ_{n+1})
            
            f_new = np.array([phi_new, p_new])

        elif method.lower() == 'leapfrog':
            # Leapfrog method (symplectic integrator)
            # Also called Störmer-Verlet or velocity Verlet
            # Momenta are computed at half-integer time steps
            
            # Half step for momentum
            p_half = f_old[1] - 0.5 * dt * np.sin(f_old[0])
            # Full step for position using half-step momentum
            phi_new = f_old[0] + dt * p_half
            # Half step for momentum to complete the step
            p_new = p_half - 0.5 * dt * np.sin(phi_new)
            
            f_new = np.array([phi_new, p_new])

        elif method.lower() == 'verlet':
            # Verlet method (symplectic integrator)
            # Standard velocity Verlet algorithm
            # Equivalent to leapfrog but computed differently
            
            # Compute force at current position
            force_old = -np.sin(f_old[0])
            
            # Update position
            phi_new = f_old[0] + dt * f_old[1] + 0.5 * dt**2 * force_old
            
            # Compute force at new position
            force_new = -np.sin(phi_new)
            
            # Update momentum using average of old and new forces
            p_new = f_old[1] + 0.5 * dt * (force_old + force_new)
            
            f_new = np.array([phi_new, p_new])
            
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'euler', 'rk4', 'trapezoidal', "
                "'drift-kick', 'leapfrog', or 'verlet'."
            )
        
        phi_new, p_new = f_new
        
        # Wrap angle to range [-4, 4] ≈ [-π, π] to keep it bounded
        # This is periodic since the pendulum physics is 2π-periodic
        if phi_new > 4:
            phi_new -= 2.0 * np.pi
        if phi_new < -4:
            phi_new += 2.0 * np.pi
        
        # Store new values in solution arrays
        phi[i] = phi_new
        p[i] = p_new
    # Display the solution as an animation
    animate_solution(t, phi, p, num_steps, animation_time)


def animate_solution(t, phi, p, num_steps, animation_time):
    """Create an animated visualization of the pendulum solution.
    
    Args:
        t: Time array
        phi: Angle array φ(t)
        p: Angular momentum array p(t)
        num_steps: Number of time steps (used to scale animation speed)
        animation_time: Total animation duration in seconds
        
    Displays three subplots:
    - Phase space trajectory (φ, p) with energy contours
    - Physical pendulum animation
    - Energy E(t) vs time to check conservation
    """
    
    def energy(phi, p):
        """Calculate total mechanical energy of the pendulum.
        
        E = (1/2)p² + (1 - cos(φ))
        where kinetic energy is (1/2)p² and potential energy is (1 - cos(φ))
        """
        return 0.5 * p**2 + 1.0 - np.cos(phi)

    # Calculate energy at all time points
    energy_array = energy(phi, p)
    energy_variation = np.max(energy_array) - np.min(energy_array)
    print(f"Energy variation (max - min): {energy_variation:.6f}")
    print("Note: Euler method does not conserve energy well.")
    print("RK4 provides better accuracy. Trapezoidal, drift-kick, leapfrog, and Verlet")
    print("are symplectic methods with excellent long-term energy conservation.")
 
    # Set up figure with three subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Subplot 1: Phase space with energy contours
    ax1 = plt.subplot(2, 2, 1)
    phi_contour, p_contour = np.meshgrid(
        np.pi * np.linspace(-1.5, 1.5, 100),
        np.linspace(-2.5, 2.5, 50)
    )
    ax1.contour(phi_contour, p_contour, energy(phi_contour, p_contour), 15)
    ax1.set_xlabel('Angle φ (rad)')
    ax1.set_ylabel('Angular Momentum p')
    ax1.set_title('Phase Space')
    phase_space_point, = ax1.plot([], [], 'ro', markersize=8, label='Current state')
    ax1.legend()

    # Subplot 2: Physical pendulum animation
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Pendulum')
    pendulum_line, = ax2.plot([], [], 'r-', linewidth=2, marker='o', 
                               markersize=15, mec='k', mfc='k', markevery=(1, 1))

    # Subplot 3: Energy vs time
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Energy E(t)')
    ax3.set_title('Energy Evolution')
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(np.min(energy_array), np.max(energy_array))
    energy_plot, = ax3.plot([], [], 'k-', linewidth=1.5)

    def update_animation(frame_index):
        """Update function for animation at each frame.
        
        Args:
            frame_index: Current frame number (time step index)
            
        Returns:
            Tuple of plot objects that were updated
        """
        # Update phase space point
        phase_space_point.set_data([phi[frame_index]], [p[frame_index]])
        
        # Update pendulum position: rod from (0,0) to (sin(φ), -cos(φ))
        pendulum_x = [0, np.sin(phi[frame_index])]
        pendulum_y = [0, -np.cos(phi[frame_index])]
        pendulum_line.set_data(pendulum_x, pendulum_y)
        
        # Update energy plot up to current time
        energy_plot.set_data(t[0:frame_index], energy_array[0:frame_index])
        
        return phase_space_point, pendulum_line, energy_plot

    # Create animation (stored globally to prevent garbage collection)
    # Scale interval with num_steps to maintain constant animation duration
    total_animation_time_ms = animation_time * 1000  # convert seconds to milliseconds
    interval = total_animation_time_ms / num_steps  # milliseconds between frames
    
    global keepMe
    keepMe = animation.FuncAnimation(
        fig, 
        update_animation, 
        frames=len(t), 
        blit=True, 
        interval=interval  # dynamically scaled interval
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the pendulum simulation
    # You can change the method to 'euler', 'rk4', or 'trapezoidal', and adjust num_steps
    # You can also adjust animation_time to control the animation duration
    pendulum(num_steps=1000, method='rk4', animation_time=60.0, initial_phi=0.0, initial_p=0.5)
