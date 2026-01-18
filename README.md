
# Computational Physics I, WS 2025/26  

## Exercise 1  

Extensive information about Python, including the language reference, tutorials, modules etc.  
can be found on [www.python.org](www.python.org) and on [docs.python.org](docs.python.org).  

---

### Some first steps in Python

- Have a look at file [loop.py](task_code\loop.py): The variable *i* is counted upwards and printed out. Try changing the loop's `range` to `range(3, 12)` or `range(3, 12, 2)`! You can also enter `range(` in Spyder's console to get an in-line hint on its meaning.  

- About Python's way of working with integer numbers: Define a variable like `num = 3` ahead of the for loop and multiply it by itself during each loop iteration. What do you observe? Use the console to divide a resulting "really large" number by 3.  

- Repeat 2., but this time, start with `num = 3.`! Can you explain the different behavior? Use Python's built-in function `type()` to see various variables' or expressions' data types.  

- Try this:  

  ```python
  c = 1.6341
  d = 9.
  d * (c / d) - c
  ```  

  Then, change `c` to `1.6346`!  

---

### Finite Differences and Convergence Test

File [convergence.py](task_code\convergence.py) contains elements for a convergence test of the right-sided operator Dt that approximates the first derivative f' as discussed in the lecture.  

- Familiarize yourself with the Python code: What does each function do? Call them from the command line!  

- A script to run an actual convergence test is in [convTest.py](task_code\convTest.py). Try it out! Which convergence order do you obtain?  
- Modify the programs to add the left-sided and central difference formulae and compare the orders of convergence.  

- What happens if you make *h* really small?  

---

## Exercise 2  

### Finite Differences and Convergence Tests II

We continue with the derivation and test of finite difference formulae:

- Verify the standard 3-point stencil for the second derivative f" as given in the lecture notes.  

- The discretizations addressed so far were based on equal spacings h between function values, employing function values f(x), f(x - h), f(x + h) etc.  

  What about a formula that uses values like $f(x)$, $f(x - h_{-})$, $f(x + h_{+})$, i.e. different spacings on either side? Can you derive, using Taylor expansion, stencils for f' and f" in this setting? Which order of accuracy do you get?  

  *Hint:* When making convergence tests, use a fixed ratio h+ / h- for the limit h+, h- → 0.  

- Derive a one-sided, second order accurate approximation for f'(x) that employs f(x), f(x - h), and f(x - 2h). Verify!  

  *Hint:* Use your previous results.  

- Richardson extrapolation is a simple way to derive higher-order finite difference formulae from lower order stencils (see the lecture notes). Use it to improve the O(h²) central difference formulae for f' and f", respectively, and verify!  

---

### Planetary Motion I

Here, we want to address the motion of a "point mass" under the influence of a central force  

$$
F(r) = - \frac{4 \pi^2 m}{r^2}
$$

Closed, elliptical trajectories are approximated in file [planet.py](task_code\planet.py) as positions of the mass at times $t_i$ with $i = 0, 1, 2, ... (N - 1)$. They are plotted together with velocity arrows, which in turn are approximated by central finite differences.  

a) Compute the particle's acceleration $a = \ddot{r}$ with finite differences and plot the respective vector arrows.  

b) Compute $|F(t_i)/m - a_i|$ for each $t_i$ and analyze the way it behaves with varying time spacings Δt and eccentricities E.  

Which error sources come into play when computing r, $\dot{r}$, and $\ddot{r}$? How can they be quantified or alleviated? Could root-finding methods help here?  

---

## Exercise 3  

### Planetary Motion II

i) Finalize last week's program for the planetary motion as discussed in the lecture: Add the computation of the exact acceleration F/m to quantify the errors for different N and E.  

*Note:* You'll find an updated version of [planet.py](task_code\planet.py) in Moodle (be sure to back up your previous one before downloading!).  

ii) Then, replace the approximate solution of Kepler's equation,  

$$
M = E - \varepsilon \sin E
$$
  
by "true" root-finding with Newton's method.

*Hint:* Copy code from file [findroot.py](task_code\findroot.py) into a new function `keplerNewton(t, ex)` inside [planet.py](task_code\planet.py).  

---

### Root-Finding Using the Secant Method

Bi-sectioning and Newton's method have been implemented in [findroot.py](task_code\findroot.py) and discussed in the lecture: Add the secant method as a further possibility when f' is not available analytically. Compare with the previous two!  

---

## Exercise 4  

### Euler's Method for the Decay Equation

File [decay.py](task_code\decay.py) approximates the solution to the simple decay equation  

$$
\frac{df(t)}{dt} = -\gamma f(t) =: G(f(t),t)
$$

with $\gamma > 0$ using the forward Euler method, i.e.,  

$$
f_{n+1} = f_n + \Delta t \, G(f_n, t) = f_n - \gamma \Delta t \, f_n
$$

(see lecture notes).  

i) Study the solution behavior for different time steps Δt: Can you verify the different regimes of the iteration eigenvalue  

$$
\lambda = \frac{f_{n+1}}{f_n} = 1 - \gamma \Delta t
$$
  
Which exact eigenvalue  

$$
\lambda_{\text{exact}} = \frac{f(t+\Delta t)}{f(t)}
$$

does the differential equation give, and how is it related to the numerical λ?  

ii) Alternatively, implement the backward Euler scheme,  

$$
f_{n+1} = f_n - \gamma \Delta t \, f_{n+1}
$$

and the trapezoidal scheme,  

$$
f_{n+1} = f_n - \frac{\gamma \Delta t}{2} \,(f_n + f_{n+1})
$$

and compare. What are the respective iteration eigenvalues here?  

---

### Euler's Method for the Harmonic Oscillator

In a similar way, study the ODE system of the harmonic oscillator,  

$$
\dot{q}(t) = \omega p(t), \quad \dot{p}(t) = -\omega q(t)
$$

as prepared in file [harmosc.py](task_code\harmosc.py) (again, see the lecture notes).  

---

## Exercise 5  

### Pendulum

a) For a pendulum with given physical parameters $m, l, g$, write down the equations of motion and an expression for the energy $E$. Explain the normalization procedure to get  

$$
\dot{\theta} = p, \quad \dot{p} = -\sin(\theta).
$$

What are the "units" of time, energy and angular momentum in this normalized form?  

b) In [pendulum.py](task_code\pendulum.py), the normalized equations are integrated using the forward-Euler method. Compare the behavior for different time steps Δt.  

c) Implement alternative methods and investigate the behavior for different Δt again:  

i) the 4th order Runge-Kutta method.  

ii) the trapezoidal rule.  

*Hint:* To get an easy implementation, introduce  

$$
\theta^* := \theta_n + \Delta t \, p_n, \quad p^* := p_n - \Delta t \, \sin(\theta_n).
$$

Plug this into the discretized equations of motion to get a single transcendent equation for $p_{n+1}$. Then, solve that in a separate function `solvePendel` by means of Newton-Raphson iteration.  

iii) the drift-kick / Leapfrog / Verlet method.  

---

## Exercise 6  

### Diffusion / Heat Equation

The Python program in file [heatEuler.py](task_code\heatEuler.py) approximates the heat equation as an initial value problem using the forward-Euler method.

a) Verify the method quantitatively: Replace the initial step function by  

$$
f(x, t = 0) = \sin\left(\frac{\nu 2\pi x}{L}\right)
$$  

as initial condition, where $\nu$ is some integer mode number. Then, compare the result at $t = 10$ with the well-known exact solution (watch `max(f)` for comparison).  
Does the method work properly?

b) When using a higher resolution in $x$ for improved accuracy, you'll find that you have to reduce $\Delta t$ as well in order to keep the computation stable (do so by increasing `N_sub`):  
Can you verify the stability condition  

$$
\Delta t \leq \frac{\Delta x^2}{2\kappa}?
$$

c) Prove this stability criterion by means of a *von-Neumann* analysis: Assume the discrete solution at $t_n$ to be  

$$
f_j^n = \hat{f}_k^n e^{ikj\Delta x}
$$  

for some given wave number $k$, and insert this into the Euler scheme to get $f_j^{n+1}$ (disregarding boundary effects). What can you tell about the resulting ratio  

$$
\lambda := \frac{f_j^{n+1}}{f_j^n} = \frac{\hat{f}_k^{n+1}}{\hat{f}_k^n}
$$  

i.e. the iteration eigenvalue, for different $\Delta t$?

d) The program employs Dirichlet conditions by keeping `f[0]` and `f[-1]` (last element in the Python array) unchanged. Try to implement:  

i) homogeneous von-Neumann conditions, $\partial_x f = 0$, at $x = 0$ and $x = L$, and/or  
ii) periodic boundary conditions, $f(x + L) = f(x)$.  

Which physical situations might these conditions reflect?

---

## Exercise 7  

### Diffusion Equation II

We consider the diffusion equation  

$$
\partial_t f = \nabla \cdot \left( \kappa \nabla f \right)
$$

a) Consider the one-dimensional setup with a diffusion coefficient that depends on position, i.e., $\kappa(x)$, and modify the forward Euler implementation accordingly. Use a conservative discretization based on the flux density  

$$
\bar{q} = -\kappa \nabla f
$$  

at “staggered” positions $x_{j+1/2}$.  
Do the results match your physical expectations?

b) You find a Python implementation of the backward Euler method for the case of constant $\kappa$ in [heatBackward.py](task_code\heatBackward.py). Improve its accuracy to $\mathcal{O}(\Delta t^2)$ by using the trapezoidal rule for integration.  
Can you estimate and verify the stability property for this case?  
Can you implement Dirichlet-type boundary conditions?

c) Generalize the implicit method from b) to $\kappa(x)$ as in a):  
What does the discretization matrix look like?

---

## Exercise 8  

### Sound Waves, 1-dimensional

The equations of ideal, adiabatic gas dynamics are:

- **Continuity equation**:  
  $$
  \frac{\partial \rho}{\partial t} + \mathbf{v} \cdot \nabla \rho = -\rho \nabla \cdot \mathbf{v}
  $$

- **Momentum equation**:  
  $$
  \rho \left( \frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v} \right) = -\nabla p
  $$

- **Energy equation**:  
  $$
  \frac{\partial p}{\partial t} + \mathbf{v} \cdot \nabla p = -\gamma p \nabla \cdot \mathbf{v}
  $$

Here, $\gamma$ is the adiabatic index of the gas, and the equations are written in advection form.

Assuming all quantities depend only on one spatial coordinate $x$ and time $t$, and considering only the longitudinal component $v_x(x, t)$, the system simplifies. Linearizing around a constant background density $\rho_0$, pressure $p_0$, and a gas at rest ($\mathbf{v}_0 = 0$), we obtain two coupled equations for the perturbations $p_1(x, t) := p(x, t) - p_0$ and $v_x(x, t)$:

$$
\frac{\partial v_x}{\partial t} = -\frac{1}{\rho_0} \frac{\partial p_1}{\partial x}, \quad \frac{\partial p_1}{\partial t} = -\gamma p_0 \frac{\partial v_x}{\partial x}
$$

a) Show that this system admits traveling-wave solutions of the form  
$v_x(x, t) = V(x - ct)$, $p_1(x, t) = P(x - ct)$.  
What is the relationship between the envelope functions $V$ and $P$?  
What is the sound velocity $c$?

b) Solve the system numerically with periodic boundary conditions in $x$. Use an isolated pressure pulse and $v_x(x, t = 0) = 0$ as initial conditions (see [sound.py](task_code\sound.py)).  
You may use Leap-Frog integration between $v_x$ and $p_1$, or apply a Lax-Wendroff or Runge-Kutta scheme.  
What do you observe?

c) Can you change the initial conditions for an isolated perturbation to move only to the left or only to the right?

---

## Exercise 9  

### Fourier Spectral Method for PDEs

The file [pdeSpectral.py](task_code\pdeSpectral.py) contains a program to solve the diffusion equation  

$$
\partial_t f = \kappa \, \partial_{xx} f
$$  

using the Fourier spectral method as discussed in the lecture.

a) Try it out and extend it to the convection-diffusion equation  

$$
\partial_t f = -c \, \partial_x f + \kappa \, \partial_{xx} f
$$  

with constant $c$.

b) Can you, in a similar way, integrate the time-dependent, free-space Schrödinger equation  

$$
i\hbar \, \partial_t \psi(x,t) = -\frac{\hbar^2}{2m} \, \partial_{xx} \psi(x,t)
$$  

in this one-dimensional setup? Try, for example, a wave packet  

$$
\psi(x, t = 0) = e^{ik_0 x} \exp\left( -\frac{1}{2} \left( \frac{x - x_c}{\sigma} \right)^2 \right)
$$  

with $x_c = 3$, $L = 10$, $k_0 = 20 \cdot \frac{2\pi}{L}$, $\sigma = \frac{1}{4}$ as initial condition, and choose $\kappa = \frac{\hbar}{2m} = \frac{1}{20}$.  
*Hints:* Start a new program file for this problem. Plot $\text{Re}(\psi)$ and $|\psi|^2$ versus $x$.

**Bonus question I:** If you were to introduce an additional potential $V(x)$ in Schrödinger’s equation – could you still treat the problem using the Fourier method?

**Bonus question II:** Which other methods could you envisage to use instead of the Fourier spectral approach?

## Emails

📧 [juergen.dreher@rub.de](juergen.dreher@rub.de)  
📧 [kevin.schoeffler@rub.de](kevin.schoeffler@rub.de)  

---
