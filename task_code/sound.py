import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Integrate up to t_end int N_cycle separate cycles
t_end = 10
N_cycle = 10
# Use N_sub substeps in each cycle, to make dt smaller than t_end/N_cycle
N_sub = 100
dt = t_end / (N_cycle * N_sub )
print('dt =', dt)  

# Domain in x
Lx = 10
Nx = 151
x = np.linspace(0., Lx*(Nx-1)/Nx, Nx)
dx = x[1] - x[0]
print('dx =', dx)  

# Some initial values for f at t=0
v = .05*np.exp( -5.*(x - 0.5*Lx)**2 )
p = .1*np.exp( -5.*(x - 0.5*Lx)**2 )

# Build a list with snapshots of f, start with initial value
v_all = np.zeros( [ N_cycle+1, Nx ] ); v_all[0, :] = v
p_all = np.zeros( [ N_cycle+1, Nx ] ); p_all[0, :] = p

# Parameters: background density and pressure, adiabatic index
rho0 = 1.5
p0 = 1.5/1.4
gamma = 1.4
cs = np.sqrt( p0*gamma/rho0)
print('cs =', cs )  
print('cfl =', dt/dx*cs)
 
for cycle in range(0, N_cycle):
    for step in range(0, N_sub):
 
        # Forward Euler / FTCS
        vnew = v - 0.5*dt/dx/rho0*( np.roll(p, -1) - np.roll(p, 1) )
        pnew = p - 0.5*dt/dx*gamma*p0*( np.roll(v, -1) - np.roll(v, 1) )
 
        v = vnew
        p = pnew
    v_all[cycle+1, :] = vnew
    p_all[cycle+1, :] = pnew
    
fig = plt.figure()
ax1 = plt.axes()
ax1.set_ylim(-.05, .15)
ax1.set_xlim(0., Lx)
ax1.set_xlabel("x")
ax1.set_ylabel("f(x, t)")

lineV, = ax1.plot([], [], '.-b', lw=2)
lineP, = ax1.plot([], [], '.-r', lw=2)
ax1.legend( [ 'vx', 'p1' ] )

def animate(i):
    ax1.set_title( f't = {i*dt*N_sub:.3f}' )
    lineV.set_data( x, v_all[i, :] )
    lineP.set_data( x, p_all[i, :] )
    return lineP, lineV

ani = animation.FuncAnimation(
    fig, animate, N_cycle+1, interval=500, blit=False, repeat=False)
