"""
Forward Euler for the Harmonic Oscillator
========================================
System:
    dq/dt =  omega * p
    dp/dt = -omega * q

This script integrates the system with the forward Euler method and compares
against the exact solution. It is structured so you can easily add alternative
methods (e.g., symplectic Euler, implicit midpoint, RK4) later.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def exact_solution(t, q0, p0, omega):
    """Analytical solution for q(t), p(t)."""
    q = q0 * np.cos(omega * t) + (p0 / omega) * np.sin(omega * t)
    p = p0 * np.cos(omega * t) - omega * q0 * np.sin(omega * t)
    return q, p


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

def forward_euler(q0, p0, omega, dt, N):
    """Explicit Euler (1st order, not energy-preserving)."""
    t = dt * np.arange(0.0, N + 1)
    q = np.zeros(N + 1)
    p = np.zeros(N + 1)
    q[0], p[0] = q0, p0

    for i in range(N):
        q[i + 1] = q[i] + dt * (omega * p[i])
        p[i + 1] = p[i] + dt * (-omega * q[i])

    return t, q, p

def backward_euler(q0, p0, omega, dt, N):
    """Implicit Euler (1st order, energy-dissipating)."""
    t = dt * np.arange(0.0, N + 1)
    q = np.zeros(N + 1)
    p = np.zeros(N + 1)
    q[0], p[0] = q0, p0

    for i in range(N):
        factor = 1 / (1 + omega**2 * dt**2)
        q[i+1] = factor * (q[i] + dt * omega * p[i])
        p[i+1] = factor * (p[i] - dt * omega * q[i])

    return t, q, p

def trapezoidal(q0, p0, omega, dt, N):
    """Trapezoidal rule (2nd order, energy-preserving)."""
    t = dt * np.arange(0.0, N + 1)
    q = np.zeros(N + 1)
    p = np.zeros(N + 1)
    q[0], p[0] = q0, p0

    for i in range(N):
        factor = 1 / (1 + (omega*dt/2)**2)
        q[i+1] = factor * ((1 - (omega*dt/2)**2) * q[i] + dt * omega * p[i])
        p[i+1] = factor * ((1 - (omega*dt/2)**2) * p[i] - dt * omega * q[i])

    return t, q, p


# Placeholders for future upgrades (e.g., symplectic Euler, implicit midpoint)
# def symplectic_euler(...):
#     pass


# ---------------------------------------------------------------------------
# Phase Difference Calculation
# ---------------------------------------------------------------------------

def compute_phase_difference(t, q, p, omega):
    """
    Compute the phase angle phi(t) from the (q, p) trajectory.
    
    For a harmonic oscillator: q(t) = A*cos(omega*t + phi)
                              p(t) = -A*omega*sin(omega*t + phi)
    
    We recover phi from: phi(t) = atan2(-p/omega, q) + omega*t
    (adjusting for the exact oscillator phase to isolate numerical phase error)
    
    Returns:
        phase: array of phase values at each time step
        phase_diff: array of phase differences (numerical - exact)
    """
    # Compute amplitude and phase from (q, p)
    A_num = np.sqrt(q**2 + (p / omega)**2)  # Amplitude
    phase_num = np.arctan2(-p / omega, q)    # Phase at each time step
    
    # Exact phase should advance linearly: phi_exact(t) = omega * t
    # But we want to extract the *deviation* from exact oscillation
    # So we compute: phase_numerical = atan2(-p/omega, q)
    # and compare to: phase_exact = atan2(-p_exact/omega, q_exact)
    
    return A_num, phase_num

# ---------------------------------------------------------------------------
# Main Integration and Analysis Function
# ---------------------------------------------------------------------------

def run_harmonic_oscillator(omega=1.0, t_end=10.0, N=20, q0=1.0, p0=0.0, 
                            calc_case='forward_euler', show_plot=True):
    """
    Solve the harmonic oscillator system with selectable integrator.

    Parameters:
        omega (float): Angular frequency
        t_end (float): Total integration time
        N (int): Number of time intervals
        q0 (float): Initial position
        p0 (float): Initial momentum
        calc_case (str): Integration method to use:
                         'forward_euler', 'backward_euler', or 'trapezoidal'
        show_plot (bool): Whether to display plots

    Returns:
        dict: Dictionary with keys 't', 'q', 'p', 'q_exact', 'p_exact',
              'energy', 'energy_exact', and error diagnostics
    """

    dt = t_end / N

    print("=" * 60)
    print("Harmonic Oscillator Integration")
    print("=" * 60)
    print(f"Method: {calc_case}")
    print(f"omega = {omega}")
    print(f"N = {N}")
    print(f"dt = {dt:.6f}  (omega*dt = {omega*dt:.6f})")
    print("=" * 60)

    # Select integrator based on calc_case using match-case
    match calc_case.lower():
        case 'forward_euler':
            t, q_num, p_num = forward_euler(q0, p0, omega, dt, N)
            method_label = 'Forward Euler'
        case 'backward_euler':
            t, q_num, p_num = backward_euler(q0, p0, omega, dt, N)
            method_label = 'Backward Euler'
        case 'trapezoidal':
            t, q_num, p_num = trapezoidal(q0, p0, omega, dt, N)
            method_label = 'Trapezoidal'
        case _:
            raise ValueError(f"Unknown calc_case: {calc_case}. Use 'forward_euler', 'backward_euler', or 'trapezoidal'.")

    # Compute exact solution
    q_ex, p_ex = exact_solution(t, q0, p0, omega)

    # Compute energies
    energy_num = 0.5 * (q_num**2 + p_num**2)
    energy_ex = 0.5 * (q_ex**2 + p_ex**2)

    # Compute errors
    err_q_final = np.abs(q_num[-1] - q_ex[-1])
    err_p_final = np.abs(p_num[-1] - p_ex[-1])
    energy_drift_final = energy_num[-1] - energy_ex[-1]
    energy_drift_rel = energy_drift_final / energy_ex[0]
    
    # Compute phase information
    A_num, phase_num = compute_phase_difference(t, q_num, p_num, omega)
    A_ex, phase_ex = compute_phase_difference(t, q_ex, p_ex, omega)
    
    # Phase difference at final time
    phase_diff_final = phase_num[-1] - phase_ex[-1]
    # Normalize to [-pi, pi]
    phase_diff_final = np.arctan2(np.sin(phase_diff_final), np.cos(phase_diff_final))

    print(f"\nFinal errors at t = {t_end}:")
    print(f"  |q_num - q_exact| = {err_q_final:.6e}")
    print(f"  |p_num - p_exact| = {err_p_final:.6e}")
    print(f"  Energy drift (abs) = {energy_drift_final:.6e}")
    print(f"  Energy drift (rel) = {energy_drift_rel:.6e}")
    print(f"\nPhase analysis at t = {t_end}:")
    print(f"  Amplitude (numerical) = {A_num[-1]:.6e}")
    print(f"  Amplitude (exact)     = {A_ex[-1]:.6e}")
    print(f"  Phase (numerical)     = {phase_num[-1]:.6e} rad")
    print(f"  Phase (exact)         = {phase_ex[-1]:.6e} rad")
    print(f"  Phase difference      = {phase_diff_final:.6e} rad")

    # Plot if requested
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

        # q(t)
        ax1.plot(t, q_num, 'rx-', label=f'q ({method_label})')
        ax1.plot(t, q_ex, 'b-', label='q (Exact)')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('q(t)')
        ax1.set_title(f'Harmonic Oscillator: q(t) - {method_label}')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Energy
        ax2.plot(t, energy_num, 'g*-', label=f'Energy ({method_label})')
        ax2.plot(t, energy_ex, 'k--', label='Energy (Exact)')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy over time')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Phase difference
        phase_diff = phase_num - phase_ex
        # Normalize to [-pi, pi]
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
        ax3.plot(t, phase_diff, 'mo-', linewidth=2, markersize=5, label='Phase difference')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('Phase difference (rad)')
        ax3.set_title(f'Phase Lag: {method_label} vs Exact')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')

        plt.tight_layout()
        plt.show()

    # Return results as dictionary
    return {
        't': t,
        'q': q_num,
        'p': p_num,
        'q_exact': q_ex,
        'p_exact': p_ex,
        'energy': energy_num,
        'energy_exact': energy_ex,
        'amplitude': A_num,
        'amplitude_exact': A_ex,
        'phase': phase_num,
        'phase_exact': phase_ex,
        'err_q': err_q_final,
        'err_p': err_p_final,
        'energy_drift_abs': energy_drift_final,
        'energy_drift_rel': energy_drift_rel,
        'phase_diff_final': phase_diff_final,
        'method': method_label,
    }


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Example: Run with Forward Euler
    result_fe = run_harmonic_oscillator(
        omega=1., t_end=10.0, N=200, q0=1.0, p0=0.0,
        calc_case='trapezoidal', show_plot=True
    )