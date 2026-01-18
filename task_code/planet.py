import numpy as np
import matplotlib.pyplot as plt

def planet(N, ex, maxIter, eps, calc_case):
    """
    Simulate and visualize planetary motion under central gravitational force.
    
    Computes the orbital trajectory of a point mass in a gravitational field
    F(r) = -4π² m / |r|². The orbit is calculated at N discrete time points,
    and numerical derivatives are used to compute velocities and accelerations.
    The error between numerical and exact accelerations is analyzed.
    
    Two plots are generated:
    1. Orbit trajectory with velocity and acceleration vectors
    2. Numerical error in acceleration vs. time
    
    Parameters:
    -----------
    N : int
        Number of time steps (points along trajectory)
        Higher N gives better accuracy but slower computation
    ex : float
        Orbital eccentricity in [0, 1)
        0 = perfect circle, 0.5 = moderately elliptical, →1 = highly eccentric
    maxIter : int
        Maximum iterations for Kepler equation solver
    eps : float
        Convergence tolerance for Kepler equation solver
    calc_case : str
        Method to solve Kepler's equation: "expansion", "newton", or "bisect"
    
    Returns:
    --------
    int
        Status code (0 for success)
        
    Notes:
    ------
    - Time is normalized to orbital period (t ∈ [0, 1))
    - Distances normalized to semi-major axis
    - Uses central finite differences for derivatives:
      * Velocity: v_i ≈ (r_{i+1} - r_{i-1})/(2Δt)
      * Acceleration: a_i ≈ (r_{i+1} - 2r_i + r_{i-1})/Δt²
    - Error sources: finite difference truncation, Kepler equation approximation
    """

    # Time interval between two time steps, normalized to the period of rotation
    delta_t = 1.0 / N
    
    # Array with points in time in [0.0, 1.0)
    t = delta_t * np.arange(N)
    
    # Calculate approximated positions (x, y) from times and eccentricity
    # using the Kepler function (see below)
    # x, y = kepler(t, ex)
    x, y = kepler(t, ex, maxIter, eps, calc_case)

    # Compute velocity v = dr/dt using central finite differences
    # v(t_i) ≈ [r(t_{i+1}) - r(t_{i-1})] / (2Δt)
    vx = (np.roll(x, -1) - np.roll(x, 1)) / (2.0 * delta_t)
    vy = (np.roll(y, -1) - np.roll(y, 1)) / (2.0 * delta_t)

    # Compute acceleration a = d²r/dt² using central finite differences
    # a(t_i) ≈ [r(t_{i+1}) - 2r(t_i) + r(t_{i-1})] / Δt²
    ax = (np.roll(x, -1) - 2.0 * x + np.roll(x, 1)) / delta_t**2
    ay = (np.roll(y, -1) - 2.0 * y + np.roll(y, 1)) / delta_t**2 
    
    # Calculate exact gravitational acceleration F/m = -4π²r/|r|³
    # This provides the "true" acceleration for comparison with numerical estimates
    r = np.sqrt(x**2 + y**2)  # Distance from central star
    F_magnitude = 4.0 * np.pi**2 / r**2  # Magnitude of gravitational force per unit mass
    
    # Decompose acceleration into Cartesian components
    # Direction: -r̂ = -(x, y)/|r| (toward central star)
    Fx_over_m = -F_magnitude * x / r  # Exact x-component of acceleration
    Fy_over_m = -F_magnitude * y / r  # Exact y-component of acceleration
    
    # Calculate numerical error: |a_exact - a_numerical|
    # This quantifies accuracy of finite difference approximations
    x_err = np.abs(Fx_over_m - ax)  # Error in x-component
    y_err = np.abs(Fy_over_m - ay)  # Error in y-component
    err = np.sqrt(x_err**2 + y_err**2)  # Total error magnitude
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)  # Upper plot: trajectory
    ax2 = fig.add_subplot(212)  # Lower plot: error

    # Upper plot: Trajectory and velocity/acceleration vectors
    ax1.plot(x, y, marker='.', color='r', linestyle='', label='Orbit trajectory')
    ax1.scatter(0.0, 0.0, marker='*', color='y', s=200, zorder=5, label='Central star')
    ax1.quiver(x, y, vx, vy, units='width', color='k', alpha=0.6, label='Velocity')
    ax1.quiver(x, y, ax, ay, units='width', color='b', alpha=0.6, label='Acceleration')
    
    ax1.set_xlabel('x', fontsize=16)
    ax1.set_ylabel('y', fontsize=16)
    ax1.set_title(f'Planetary Motion (N={N}, eccentricity={ex})', fontsize=14)
    ax1.set_aspect('equal')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Lower plot: Error measure vs. time
    ax2.plot(t, err, color='k', linewidth=2, label='Total error')
    ax2.plot(t, x_err, color='r', alpha=0.7, label='Error along x')
    ax2.plot(t, y_err, color='b', alpha=0.7, label='Error along y')
    
    ax2.set_xlabel('t (normalized time)', fontsize=16)
    ax2.set_ylabel('|F/m - a|', fontsize=16)
    ax2.set_title('Numerical Error in Acceleration', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return 0

def kepler(t, ex, maxIter, eps, calc_case):
    """
    Calculate Cartesian coordinates of a Kepler ellipse by solving Kepler's equation.
    
    This function computes positions on an elliptical orbit by first solving
    Kepler's equation M = E - ex*sin(E) for the eccentric anomaly E, then
    converting to true anomaly and finally to Cartesian coordinates.
    
    Parameters:
    -----------
    t : ndarray
        Array of times normalized to the orbital period, t ∈ [0, 1)
    ex : float
        Orbital eccentricity in [0, 1)
        0 = circular orbit, approaching 1 = parabolic (open) orbit
    maxIter : int
        Maximum number of iterations for root-finding algorithms
    eps : float
        Convergence tolerance for root-finding algorithms
    calc_case : str
        Method to solve Kepler's equation:
        - "expansion": Series expansion up to O(ex³)
        - "newton": Newton-Raphson method
        - "bisect": Bisection method
    
    Returns:
    --------
    x, y : ndarray
        Cartesian position vectors with same length as t
        Normalized to the semi-major axis
    
    Notes:
    ------
    Algorithm steps:
    1. Compute mean anomaly M = 2π*t
    2. Solve Kepler's equation for eccentric anomaly E
    3. Convert E to true anomaly (polar angle) T
    4. Compute orbital radius r from ellipse equation
    5. Transform to Cartesian coordinates (x, y)
    
    References:
    -----------
    Kepler's laws: http://en.wikipedia.org/wiki/Kepler's_laws_of_planetary_motion
    """

    # Mean anomaly: angle that increases linearly with time
    # M = 0 at perihelion (closest approach), M = 2π after one complete orbit
    M = 2.0 * np.pi * t

    # Solve Kepler's equation M = E - ex*sin(E) for eccentric anomaly E
    # Choose solution method based on calc_case parameter
    match calc_case:
        case "expansion":
            E = M + (ex - ex**3 / 8.0) * np.sin(M) + \
            ex**2 / 2.0 * np.sin(2 * M) + \
            3.0 / 8.0 * ex**3 * np.sin(3 * M)
        case "newton":
            E = newton(M, ex, maxIter, eps)
        case "bisect":
            E = bisect(M, ex, maxIter, eps)
        case _:
            raise ValueError(f"Unknown calc_case '{calc_case}'")

    # Calculate true anomaly T (actual polar angle in orbit)
    # Transformation from eccentric anomaly E to true anomaly T
    T = 2.0 * np.arctan(np.sqrt((1.0 + ex) / (1.0 - ex)) * np.tan(E / 2.0))

    # Calculate orbital radius from ellipse equation
    # r = a(1-e²)/(1+e*cos(T)) where a=1 (normalized to semi-major axis)
    r = (1.0 - ex**2) / (1.0 + ex * np.cos(T))
    
    # Convert polar coordinates (r, T) to Cartesian (x, y)
    # Origin at focus (central star position)
    x = r * np.cos(T)
    y = r * np.sin(T)
    
    return x, y

def newton(M, ex, maxIter, eps):
    """
    Solve Kepler's equation using Newton-Raphson iteration (vectorized).
    
    Solves the transcendental equation M = E - ex*sin(E) for the eccentric
    anomaly E using Newton's method with numerical derivatives.
    
    Algorithm:
    ----------
    For each iteration:
        f(E) = E - ex*sin(E) - M
        f'(E) = 1 - ex*cos(E)
        E_new = E - f(E)/f'(E)
    
    Converges when max|f(E)| < eps
    
    Parameters:
    -----------
    M : ndarray
        Mean anomaly values (M = 2π*t for normalized time)
    ex : float
        Orbital eccentricity in [0, 1)
    maxIter : int
        Maximum number of iterations allowed
    eps : float
        Convergence tolerance (residual threshold)
    
    Returns:
    --------
    E : ndarray
        Eccentric anomaly values (same shape as M)
        
    Raises:
    -------
    Exception
        If convergence is not achieved within maxIter iterations
        
    Notes:
    ------
    - Vectorized implementation: operates on entire M array simultaneously
    - Initial guess: E = M (good for low eccentricity)
    - Convergence is typically quadratic (doubles precision each iteration)
    """

    def f(E):
        return E - M - ex*np.sin(E)
    
    def fPrime(E):
        return 1-ex*np.cos(E)
    
    E = M.copy()

    for _ in range(maxIter):
        fE = f(E)

        if np.max(np.abs(fE)) < eps:
            return E
        
        E = E - fE/fPrime(E)

    else:
        raise Exception(f"Couldn't get close to root in {maxIter} iterations." )
    
def bisect(M, ex, maxIter, eps):
    """
    Solve Kepler's equation using the bisection method (element-wise).
    
    For each element in M, finds the root of f(E) = E - ex*sin(E) - M
    using the bisection algorithm on the interval [0, 2π].
    
    Algorithm:
    ----------
    For each M[i]:
        1. Start with bracket [a, b] = [0, 2π] where f(a)*f(b) < 0
        2. Compute midpoint xm = (a + b)/2
        3. Evaluate f(xm)
        4. If |f(xm)| < eps: solution found
        5. Else: replace a or b with xm to maintain sign change
        6. Repeat until convergence or maxIter reached
    
    Parameters:
    -----------
    M : ndarray
        Mean anomaly values (M = 2π*t for normalized time)
    ex : float
        Orbital eccentricity in [0, 1)
    maxIter : int
        Maximum iterations per element
    eps : float
        Convergence tolerance |f(E)| < eps
    
    Returns:
    --------
    E : ndarray
        Eccentric anomaly values (same shape as M)
        
    Raises:
    -------
    ValueError
        If initial bracket doesn't satisfy f(a)*f(b) < 0
    Exception
        If convergence not achieved within maxIter iterations
        
    Notes:
    ------
    - Solves each element independently (not vectorized)
    - Guaranteed to converge if root exists in [0, 2π]
    - Convergence is linear (error halves each iteration)
    - More robust but slower than Newton's method
    """
    
    def f(E, M):
        return E - M - ex*np.sin(E)
    
    E = np.zeros_like(M)

    for idx, m in enumerate(M):
        a, b = 0.0, 2.0*np.pi
        fa, fb = f(a, m), f(b, m)

        if fa * fb > 0:
            raise ValueError('f(a) and f(b) must have opposite signs')

        for _ in range(maxIter):
            xm = (a + b) / 2.0
            fm = f(xm, m)

            if np.abs(fm) < eps:
                E[idx] = xm
                break
            
            if( fm * fa > 0 ):
                a = xm
                fa = fm
            elif( fm * fb > 0 ):
                b = xm
                fb = fm
            else:
                raise ValueError("Code fucked up")
     
        else:
            raise Exception(f"Couldn't get close to root in {maxIter} iterations.")
        
    return E

if __name__ == '__main__':
    # Execute: Call function 'planet' with appropriate arguments for N and ex
    planet(20, .5, 100, 1e-15, "newton")