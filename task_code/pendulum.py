import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# create the numerical solution
def pendulum():
    t = np.linspace(0., 60., 201)
    phi = 0*t # to be filled
    p = 0*t # to be filled
    
    # Initial Conditions
    phi[0] = 0.0
    p[0] = .2
    
    def G(f): # RHS of pendulum equations
        return np.array( [ f[1], -np.sin( f[0] ) ] )

    # compute solution to phi dot = p and p dot = -sin(phi) into arrays phi and p
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        phi_old = phi[i-1]; p_old = p[i-1] # pick old values from array

        # Euler step
        f = [ phi_old, p_old ] # collect (phi, p) into f
        phi_new,p_new = f + dt*G( f ) # Euler step
        
        # Limit phi to range of [-4, 4]
        if (phi_new > 4): phi_new = phi_new - 2.*np.pi
        if (phi_new < -4): phi_new = phi_new + 2.*np.pi
        
        # put new values into arrays
        phi[i] = phi_new
        p[i] = p_new
    # show solution
    show(t, phi, p)
    
# display the solution as animation
def show(t, phi, p):
    def energy(phi, p):
        return 0.5*p**2 + 1. - np.cos(phi)

    en = energy(phi, p)
    print("Energy max-min", np.max(en) - np.min(en) )
 
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1) # axes for phase space plot
    phi_cont, p_cont = np.meshgrid(np.pi*np.linspace(-1.5, 1.5, 100), np.linspace(-2.5, 2.5, 50))  
    ax1.contour(phi_cont, p_cont, energy(phi_cont, p_cont), 15);
    ax1.set_xlabel('phi'); ax1.set_ylabel('p');
    psPlot, = ax1.plot([], [], 'ro')

    ax2 = plt.subplot(2, 2, 2) # axes for pendulum sketch
    ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2); ax2.set_aspect('equal')
    ax2.set_xticks([]); ax2.set_yticks([])
    penPlot, = ax2.plot( [], [], c='r', marker='o', mec='k', mfc='k', markevery=(1, 1))

    ax3 = plt.subplot(2, 2, 3) # axes for energy plot
    ax3.set_xlabel('t'); ax3.set_ylabel('E(t)');
    ax3.set_xlim( t[0], t[-1] ); ax3.set_ylim( np.min(en), np.max(en) )
    enPlot, = ax3.plot( [], [], 'k-' )

    def doPlot(i):
        psPlot.set_data( [phi[i]], [p[i]] )
        penPlot.set_data( [0, np.sin(phi[i])], [0, -np.cos(phi[i])] )
        enPlot.set_data( t[0:i], en[0:i] )
        ax3.set_xlim( t[0], t[-1] ); ax3.set_ylim( np.min(en), np.max(en) )
        return psPlot, penPlot, enPlot

    global keepMe
    keepMe = animation.FuncAnimation(fig, doPlot, frames=len(t), blit='True', interval = 100 )
    plt.show()
 
# call the main program
pendulum()
