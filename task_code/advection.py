import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Integrate up to t_end int N_cycle separate cycles
t_end = 10
N_cycle = 20
# Use N_sub substeps in each cycle, to make dt smaller than t_end/N_cycle
N_sub = 30
dt = t_end / (N_cycle * N_sub )
print('dt =', dt)  

# Domain in x
Lx = 10
Nx = 151
x = np.linspace(0., Lx*(Nx-1)/Nx, Nx)
dx = x[1] - x[0]
print('dx =', dx)  

# Some initial values for f at t=0
f = np.exp( -(x - 0.5*Lx)**2 )

# Build a list with snapshots of f, start with initial value
f_all = np.zeros( [ N_cycle+1, Nx ] )
f_all[0, :] = f

# Advecction speed
c = 1.
print('c =', c)

eta = c * dt / dx
print('eta =', eta)

for cycle in range(0, N_cycle):
    for step in range(0, N_sub):
 
        # Forward Euler / FTCS
        fnew = f - 0.5*eta*( np.roll(f, -1) - np.roll(f, 1) ) # FTCS

        # Lax-Friedrichs:
        #fnew = 0.5*( np.roll(f, -1) + np.roll(f, 1) ) \
        #    - 0.5*eta*( np.roll(f, -1) - np.roll(f, 1) )

        # Upwind, left-sided
        #fnew = f - eta * ( f - np.roll(f, 1) )
        
        # Lax-Wendroff:
        #fhalf = 0.5*( np.roll(f, -1) + f ) - 0.5*eta*( np.roll(f, -1) - f )
        #fnew = f - eta*( fhalf - np.roll(fhalf, 1) ) # Lax-Wendroff, final

        f = fnew
        
    f_all[cycle+1, :] = f
    print(f'Cycle {cycle}: max f = {np.max(f)}' )
    

fig = plt.figure()
ax1 = plt.axes()
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlim(0., Lx)
ax1.set_xlabel("x")
ax1.set_ylabel("f(x, t)")
ax1.plot(x, f_all[0,:], 'r')

line, = ax1.plot([], [], '.-', lw=2)

def animate(i):
    ax1.set_title( f't = {i*dt*N_sub:.3f}' )
    line.set_data( x, f_all[i, :] )
  #  ax1.plot( x, f_all[i, :], '-b', lw=2 )
    return line,

ani = animation.FuncAnimation(
    fig, animate, N_cycle+1, interval=200, blit=False, repeat=False)
