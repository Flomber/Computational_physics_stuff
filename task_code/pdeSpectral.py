import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Build x-space array with x[0] = 0 und x[m] = L
m = 64  # number of intervals
L = 10  # domain length
x = np.linspace(0., L, m+1)

# Initial values for f
f = 1 + 0*x;
f[ (x >= 2) & (x <= 3) ] = 3
f[ (x >= 8) & (x < 9) ] = 2

# Diffusion coefficient
kappa = 0.2

# We integrate up to t_end and plot the solution at
# t = 0, 1, 2, 3, ...
tEnd = 10

# save the initial values into fSave for later plotting
fSave = [ f.copy() ]

# build k-space to cover [ -k_nyq , k_nyq )
delta_k = 2 * np.pi / L
k = delta_k * np.linspace( -m/2, m/2-1, m )
# and shift it, so that it starts with k0 = 0 (DC wave number)
# and k_nyq is in the center
k = np.fft.fftshift( k )

# take initial spectrum, but with last value from f eliminated (periodic conditions)
f_hat_0 = np.fft.fft( f[:-1] )

# The loop over the tEnd time intervals
for t in range(1, tEnd+1):

    # compute spectrum at time t from PDE
    f_hat = f_hat_0 * np.exp( - k**2 * kappa * t )

    # transform back to x-space and duplicate first element to end
    f = np.real( np.fft.ifft(f_hat) )
    f = np.append(f, f[0])

    # Append current values to fSave
    fSave.append( f.copy() )

# Plot the sequence of f-values in fPlot:
# Set up plot and create a list of empty lines
fig, ax = plt.subplots()
ax.set_xlabel('x'); ax.set_ylabel('f')
ax.set_xlim( x[0], x[-1] ); ax.set_ylim(.8, 3.2); 
lines = ( ax.plot([], [], 'r-*') ) # first (initial) line in red
for f in fSave[1:]:
    l, = ax.plot([], [], 'b-*') # all other lines in blue
    lines.append(l)
    
def doPlot(t):
    print('t =', t)
    for i in range(t+1):
        lines[i].set_data(x, fSave[i]) # set computed data up to t
    for i in range(t+1, len(fSave)):
        lines[i].set_data([], []) # set empty data after t
    return lines
    
# play the list of lists
keepMe = animation.FuncAnimation(fig, doPlot, frames=len(fSave), blit=True, interval = 500, repeat=False)
 
plt.show()
