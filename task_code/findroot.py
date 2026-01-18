import numpy as np

# Test function
def f(x):
    return np.cos(x)

# Derivative, needed in Newton's method
def fprime(x):
    return -np.sin(x)

# Bi-sectioning
def bisect(a, b, eps):
    fa = f(a)
    fb = f(b)
    if fa*fb > 0:
        raise ValueError('f(a) and f(b) must have opposite signs')
    
    maxIter = 100
    for i in range(maxIter):
        xm = ( a + b ) / 2.
        fm = f(xm)
        print("Value at ", xm, " is ", fm)

        if np.abs(fm) < eps:
            return [xm, fm, i]
        
        if( fm * fa > 0 ):
            a = xm
            fa = fm
        elif( fm * fb > 0 ):
            b = xm
            fb = fm
     
    raise Exception("Couldn't get close to root in " + str(maxIter) + " iterations." )

# Newton's method    
def newton(x, eps):
      
    maxIter = 100
    for i in range(maxIter):
        fx = f(x)
        print("Value at ", x, " is ", fx)

        if np.abs(fx) < eps:
            return [x, fx, i]
        
        # Iteration
        x = x - fx / fprime(x)
        
        
    raise Exception("Couldn't get close to root in " + str(maxIter) + " iterations." )


# Run test
tol = 1e-8
print( bisect(0., 2., tol) )
print( newton(0.9, tol) )

