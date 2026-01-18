"""
Numerical Integration Methods for the Decay Equation
=====================================================
This script compares different discretization schemes for solving:
    df/dt = -gamma * f(t)

Methods implemented and compared:
    1. Forward Euler (explicit)
    2. Backward Euler (implicit)
    3. Trapezoidal rule (implicit)

The code compares numerical solutions with the exact analytical solution
f(t) = f0 * exp(-gamma*t) and analyzes convergence and stability.

Key parameters to experiment with:
  - N: number of time intervals (larger N = smaller dt = better accuracy)
  - gamma: decay rate (determines how fast f decays)

Each method has an iteration eigenvalue lambda that determines stability.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Integration Methods (Functions)
# ============================================================================

def forward_euler(f0, gamma, dt, N):
    """
    Forward Euler method (explicit, 1st order)
    
    Formula: f_{n+1} = f_n - gamma * dt * f_n = (1 - gamma*dt) * f_n
    
    Eigenvalue: lambda = 1 - gamma*dt
    Stable for: gamma*dt <= 2
    
    Parameters:
        f0: initial value f(0)
        gamma: decay rate
        dt: time step
        N: number of intervals
    
    Returns:
        t: array of time values
        f: array of solution values
        lambda_num: numerical eigenvalue
    """
    t = dt * np.arange(0., N+1)
    f = np.zeros(N+1)
    f[0] = f0
    
    lambda_num = 1.0 - gamma * dt
    
    for i in range(N):
        f_old = f[i]
        # Discretized ODE: f_{n+1} = f_n + dt * G(f_n, t_n)
        # where G(f,t) = -gamma*f is the RHS of df/dt = G(f,t)
        f_new = f_old + dt * (-gamma * f_old)
        f[i+1] = f_new
    
    return t, f, lambda_num


def backward_euler(f0, gamma, dt, N):
    """
    Backward Euler method (implicit, 1st order)
    
    Formula: f_{n+1} = f_n - gamma * dt * f_{n+1}
    
    Rearranged: f_{n+1} = f_n / (1 + gamma*dt)
    
    Eigenvalue: lambda = 1 / (1 + gamma*dt)
    Stable for: all gamma*dt > 0 (unconditionally stable)
    
    Parameters:
        f0: initial value f(0)
        gamma: decay rate
        dt: time step
        N: number of intervals
    
    Returns:
        t: array of time values
        f: array of solution values
        lambda_num: numerical eigenvalue
    """
    t = dt * np.arange(0., N+1)
    f = np.zeros(N+1)
    f[0] = f0
    
    lambda_num = 1.0 / (1.0 + gamma * dt)
    
    for i in range(N):
        f_old = f[i]
        f_new = f_old / (1.0 + gamma * dt)
        f[i+1] = f_new

    return t, f, lambda_num


def trapezoidal_rule(f0, gamma, dt, N):
    """
    Trapezoidal rule (implicit, 2nd order)
    
    Formula: f_{n+1} = f_n - (gamma*dt/2) * (f_n + f_{n+1})
    
    Rearranged: f_{n+1} = f_n * (2 - gamma*dt) / (2 + gamma*dt)
    
    Eigenvalue: lambda = (2 - gamma*dt) / (2 + gamma*dt)
    Stable for: all gamma*dt > 0 (unconditionally stable)
    
    Parameters:
        f0: initial value f(0)
        gamma: decay rate
        dt: time step
        N: number of intervals
    
    Returns:
        t: array of time values
        f: array of solution values
        lambda_num: numerical eigenvalue
    """
    t = dt * np.arange(0., N+1)
    f = np.zeros(N+1)
    f[0] = f0
    
    lambda_num = (2.0 - gamma * dt) / (2.0 + gamma * dt)
    
    for i in range(N):
        f_old = f[i]
        f_new = (f_old - gamma*dt/2 * f_old) / (1 + gamma*dt/2)
        f[i+1] = f_new
    return t, f, lambda_num


# ============================================================================
# Main Program
# ============================================================================

# Physical and numerical parameters
gamma = 1.0             # decay rate (1/time)
t_end = 10             # total time to simulate
f0 = 1.0               # initial value f(0)
N = 100                  # number of time intervals (increase for finer resolution)

# Compute time step
dt = t_end / N

print("=" * 60)
print("Numerical Integration Methods for Decay Equation")
print("=" * 60)
print(f"Number of intervals N: {N}")
print(f"Time step dt = {dt:.6f}")
print(f"Product gamma*dt = {gamma*dt:.6f}")
print("=" * 60)

# Exact solution for reference
def exact_solution(t, f0, gamma):
    """Analytical solution: f(t) = f0 * exp(-gamma*t)"""
    return f0 * np.exp(-gamma * t)

# Compute exact eigenvalue for comparison
lambda_exact = np.exp(-gamma * dt)
print(f"Exact eigenvalue lambda_exact = exp(-gamma*dt) = {lambda_exact:.6f}\n")

# Solve using Forward Euler
print("Method 1: Forward Euler")
print("-" * 60)
t_fe, f_fe, lambda_fe = forward_euler(f0, gamma, dt, N)
print(f"  Numerical eigenvalue: {lambda_fe:.6f}")
print(f"  Stability: |lambda| = {abs(lambda_fe):.6f} {'✓ stable' if abs(lambda_fe) <= 1 else '✗ unstable'}")
f_exact_fe = exact_solution(t_fe, f0, gamma)
error_fe = np.abs(f_fe[-1] - f_exact_fe[-1])
print(f"  Error at t_end: {error_fe:.8e}")
print()

# Solve using backward Euler
print("Method 2: Backward Euler")
print("-" * 60)
t_be, f_be, lambda_be = backward_euler(f0, gamma, dt, N)
print(f"  Numerical eigenvalue: {lambda_be:.6f}")
f_exact_be = exact_solution(t_be, f0, gamma)
error_be = np.abs(f_be[-1] - f_exact_be[-1])
print(f"  Error at t_end: {error_be:.8e}")
print()

# Solve using trapezoidal Rule
print("Method 3: Trapezoidal Rule")
print("-" * 60)
t_tr, f_tr, lambda_tr = trapezoidal_rule(f0, gamma, dt, N)
print(f"  Numerical eigenvalue: {lambda_tr:.6f}")
f_exact_tr = exact_solution(t_tr, f0, gamma)
error_tr = np.abs(f_tr[-1] - f_exact_tr[-1])
print(f"  Error at t_end: {error_tr:.8e}")
print()

# ============================================================================
# Visualization
# ============================================================================

plt.figure(figsize=(12, 7))

# Exact solution for reference
t_exact = np.linspace(0., t_end, 200)
f_exact = exact_solution(t_exact, f0, gamma)
plt.plot(t_exact, f_exact, 'k-', linewidth=2.5, label='Exact solution', zorder=5)

# Forward Euler
plt.plot(t_fe, f_fe, 'ro-', linewidth=2, markersize=7, label='Forward Euler', alpha=0.8)

# Backward Euler
plt.plot(t_be, f_be, 'gs-', linewidth=2, markersize=7, label='Backward Euler', alpha=0.8)

# Trapezoidal
plt.plot(t_tr, f_tr, 'b^-', linewidth=2, markersize=7, label='Trapezoidal Rule', alpha=0.8)

# Formatting
plt.xlabel("Time $t$", fontsize=12)
plt.ylabel("Solution $f(t)$", fontsize=12)
plt.title(f"Decay Equation: Comparison of Integration Methods (N={N}, $\\Delta t$={dt:.3f})",
          fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=11)
plt.tight_layout()
plt.show()