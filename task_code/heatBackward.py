import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Build x arraywith x[0] = 0 und x[m] = L
m = 41
L = 10
x = np.linspace(0., L, m)
dx = x[1] - x[0]

# Initial values for f
f = 1 + 0*x;
f[ (x >= 2) & (x <= 3) ] = 3
f[ (x >= 8) & (x<9) ] = 2

# Diffusion coefficient
kappa = 0.3

# We integrate up to t_end and plot the solution at
# t = 0, 1, 2, 3, ...
tEnd = 10

# In each time interval of length 1, do nSub substeps
nSub = 15
# gives dt as
dt = 1. / nSub

# Abbreviate
eta = kappa * dt / dx**2
print('eta =', eta)
# save the initial values into fSave for later plotting
fSave = [ f.copy() ]

# create iteration matrix
one = np.eye(m);
D = eta* ( -2*one + np.roll(one, 1, axis=0) + \
          np.roll(one, -1, axis=0) )
R = one - D;

# The 'outer' loop over the tEnd time intervals
for t in range(tEnd):

    # The 'inner' loop over the time steps of dt
    for step in range(nSub):
        f = np.linalg.solve(R, f)
        
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
keepMe = animation.FuncAnimation(fig, doPlot, frames=len(fSave), blit=True, interval = 1000, repeat=False)
plt.show()
 
