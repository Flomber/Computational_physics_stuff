import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x) * np.sin(x)
    
def fPrime(x):
    return np.exp(-x) * ( np.cos(x) - np.sin(x) )

def fDoublePrime(x):
    return -2*np.e**(-x) * np.cos(x)

def showF(x):
    xPlot = x + np.linspace(-2.5, 2.5, 100)
    plt.plot(xPlot, f(xPlot)); plt.xlabel('x'); plt.ylabel('f(x)')
    plt.plot(x, f(x), 'r*')
    print( f"f({x}) = { f(x) }")
    print( f"f'({x}) = { fPrime(x) }")
    
# =================== Exercise 1 derivatives =================== 

def fPrimeR(x, h):
    return ( f( x + h) - f(x) ) / h

def fPrimeL(x, h):
    return ( f( x ) - f(x - h) ) / h

def fPrimeC(x, h):
    return ( f( x + h ) - f(x - h ) ) / (2*h)

def errorR(x, h):
    return np.abs( fPrime(x) - fPrimeR(x, h ))

def errorL(x, h):
    return np.abs( fPrime(x) - fPrimeL(x, h ))

def errorC(x, h):
    return np.abs( fPrime(x) - fPrimeC(x, h ))

# =================== Exercise 2 derivatives =================== 

# second derivative

def fDoublePrimeC(x, h):
    return (f(x-h) - 2*f(x) + f(x+h))/h**2

def errorDoubleC(x, h):
    return np.abs(fDoublePrime(x) - fDoublePrimeC(x, h))

# derivaives with different spacings (h_+, h_-)

def fPrimePositive(x, h_p):
    return (f(x+h_p) - f(x))/h_p

def fPrimeNegative(x, h_n):
    return (f(x) - f(x-h_n))/h_n

def fPrimeCSpacing(x, h_p, h_n):
    return (f(x+h_p) - f(x-h_n))/(h_p + h_n)

def fDoublePrimeSpacing(x, h_p, h_n):
    return 2 * (h_p*f(x-h_n) - (h_p+h_n)*f(x) + h_n*f(x+h_p))/(h_p*h_n*(h_p+h_n))

# def fDoublePrimeSpacing(x, h_p, h_n):
#     """Second derivative with unequal spacing h_p (right) and h_n (left)"""
#     return 2 * (f(x+h_p)/h_p - f(x)/(h_p*h_n) + f(x-h_n)/h_n) / (h_p + h_n)

def errorPositive(x, h_p):
    return np.abs(fPrime(x) - fPrimePositive(x, h_p))

def errorNegative(x, h_n):
    return np.abs(fPrime(x) - fPrimeNegative(x, h_n))

def errorCSpacing(x, h_p, h_n):
    return np.abs(fPrime(x) - fPrimeCSpacing(x, h_p, h_n))

def errorDoublePrimeSpacing(x, h_p, h_n):
    return np.abs(fDoublePrime(x) - fDoublePrimeSpacing(x, h_p, h_n))

# derivative with -h and -2h

def fPrimeDoubleR(x, h):
    return (3*f(x) - 4*f(x-h) + f(x-2*h))/2*h

def errorDoubleR(x, h):
    return np.abs(fPrime(x) - fPrimeDoubleR(x, h))

# Richardson extrapolation

def fPrimeRichardson(x, h):
    return (2*fPrimeC(x, h) - fPrimeC(x, 2*h))/(2-1)

def fDoublePrimeRichardson(x, h):
    return (4*fDoublePrimeC(x, h) - fDoublePrimeC(x, 2*h))/(4-1)

def errorRichardson(x, h):
    return np.abs(fPrime(x) - fPrimeRichardson(x, h))

def errorDoubleRichardson(x, h):
    return np.abs(fDoublePrime(x) - fDoublePrimeRichardson(x, h))
