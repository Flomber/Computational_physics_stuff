import numpy as np 
import convergence as co
import matplotlib.pyplot as plt

def ex_1(location: float = 2., spacing: float = 2., potency: tuple = (-30., 5.), steps: int = 20) -> None:

    x = location
    h = spacing ** np.linspace(potency[0], potency[1], steps)

    errorR = co.errorR(x, h)
    errorL = co.errorL(x, h)
    errorC = co.errorC(x, h)

    # printe Fehler und weitere Daten in die Konsole
    print("="*70)
    print(f"Convergence Test für f'({x})")
    print("="*70)
    print(f"\nExakte Ableitung: f'({x}) = {co.fPrime(x):.10f}\n")
    print(f"{'h':>12} | {'Error Right':>14} | {'Error Left':>14} | {'Error Center':>14}")
    print("-"*70)
    for i in range(len(h)):
        print(f"{h[i]:>12.6e} | {errorR[i]:>14.6e} | {errorL[i]:>14.6e} | {errorC[i]:>14.6e}")
    print("="*70)
    print("\nHinweis: Im log-log Plot zeigt die Steigung die Konvergenzordnung:")
    print("  - Right/Left: Steigung ≈ 1  → O(h¹)")
    print("  - Center:     Steigung ≈ 2  → O(h²)\n")

    # Erstelle zwei Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Konvergenztest
    ax1.loglog(h, errorR, '*-', label="right")
    ax1.loglog(h, errorL, '*-', label="left")
    ax1.loglog(h, errorC, '*-', label="center")
    ax1.set_xlabel('h')
    ax1.set_ylabel('error')
    ax1.set_title(f"Konvergenztest für f'(x) bei x={x}")
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Funktion und Ableitung
    xPlot = np.linspace(x - 2.5, x + 2.5, 200)
    ax2.plot(xPlot, co.f(xPlot), 'b-', linewidth=2, label='f(x) = exp(-x) * sin(x)')
    ax2.plot(xPlot, co.fPrime(xPlot), 'r-', linewidth=2, label="f'(x)")
    ax2.plot(x, co.f(x), 'bo', markersize=10, label=f'f({x})')
    ax2.plot(x, co.fPrime(x), 'ro', markersize=10, label=f"f'({x})")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=x, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Funktion und ihre Ableitung')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

def ex_2(location: float = 2., spacing: float = 2., potency: tuple = (-30., 5.), steps: int = 20):
    
    x = location
    h = spacing ** np.linspace(potency[0], potency[1], steps)

    # All error calculations for Exercise 2 derivatives
    # Second derivative - central difference
    errorDoubleC = co.errorDoubleC(x, h)
    
    # First derivatives with unequal spacing (using h_p = h, h_n = h for convergence test)
    errorPositive = co.errorPositive(x, h)
    errorNegative = co.errorNegative(x, h)
    errorCSpacing = np.array([co.errorCSpacing(x, hi, hi) for hi in h])
    
    # Second derivative with unequal spacing (using h_p = h, h_n = h)
    errorDoublePrimeSpacing = np.array([co.errorDoublePrimeSpacing(x, hi, hi) for hi in h])
    
    # One-sided derivative with f(x), f(x-h), f(x-2h)
    errorDoubleR = co.errorDoubleR(x, h)
    
    # Richardson extrapolation
    errorRichardson = co.errorRichardson(x, h)
    errorDoubleRichardson = co.errorDoubleRichardson(x, h)

    # Print errors to console
    print("="*100)
    print(f"Convergence Test für Exercise 2 Derivatives bei x={x}")
    print("="*100)
    print(f"\nExakte Ableitungen:")
    print(f"  f'({x})  = {co.fPrime(x):.10f}")
    print(f"  f''({x}) = {co.fDoublePrime(x):.10f}\n")
    print(f"{'h':>12} | {'Err f\' Pos':>12} | {'Err f\' Neg':>12} | {'Err f\' CSpac':>12} | {'Err f\' DblR':>12} | {'Err f\' Rich':>12} | {'Err f\'\' C':>12} | {'Err f\'\' Spac':>12} | {'Err f\'\' Rich':>12}")
    print("-"*100)
    for i in range(len(h)):
        print(f"{h[i]:>12.6e} | {errorPositive[i]:>12.6e} | {errorNegative[i]:>12.6e} | {errorCSpacing[i]:>12.6e} | {errorDoubleR[i]:>12.6e} | {errorRichardson[i]:>12.6e} | {errorDoubleC[i]:>12.6e} | {errorDoublePrimeSpacing[i]:>12.6e} | {errorDoubleRichardson[i]:>12.6e}")
    print("="*100)
    print("\nHinweis: Im log-log Plot zeigt die Steigung die Konvergenzordnung:")
    print("  First derivatives:")
    print("    - Positive/Negative:    Steigung ≈ 1  → O(h¹)")
    print("    - Central Spacing:      Steigung ≈ 2  → O(h²) (bei h+ = h-)")
    print("    - One-sided (DoubleR):  Steigung ≈ 2  → O(h²)")
    print("    - Richardson:           Steigung ≈ 4  → O(h⁴)")
    print("  Second derivatives:")
    print("    - Center:               Steigung ≈ 2  → O(h²)")
    print("    - Unequal Spacing:      Steigung ≈ 2  → O(h²) (bei h+ = h-)")
    print("    - Richardson:           Steigung ≈ 4  → O(h⁴)\n")

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: First derivatives convergence test
    ax1.loglog(h, errorPositive, '*-', label="f' Positive")
    ax1.loglog(h, errorNegative, '*-', label="f' Negative")
    ax1.loglog(h, errorCSpacing, '*-', label="f' Central Spacing")
    ax1.loglog(h, errorDoubleR, '*-', label="f' One-sided (DoubleR)")
    ax1.loglog(h, errorRichardson, '*-', label="f' Richardson")
    ax1.set_xlabel('h')
    ax1.set_ylabel('error')
    ax1.set_title(f"Konvergenztest für f'(x) bei x={x}")
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Second derivatives convergence test
    ax2.loglog(h, errorDoubleC, '*-', label="f'' Center")
    ax2.loglog(h, errorDoublePrimeSpacing, '*-', label="f'' Unequal Spacing")
    ax2.loglog(h, errorDoubleRichardson, '*-', label="f'' Richardson")
    ax2.set_xlabel('h')
    ax2.set_ylabel('error')
    ax2.set_title(f"Konvergenztest für f''(x) bei x={x}")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 3: Function and first derivative
    xPlot = np.linspace(x - 2.5, x + 2.5, 200)
    ax3.plot(xPlot, co.f(xPlot), 'b-', linewidth=2, label='f(x) = exp(-x) * sin(x)')
    ax3.plot(xPlot, co.fPrime(xPlot), 'r-', linewidth=2, label="f'(x)")
    ax3.plot(x, co.f(x), 'bo', markersize=10, label=f'f({x})')
    ax3.plot(x, co.fPrime(x), 'ro', markersize=10, label=f"f'({x})")
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(x=x, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Funktion und ihre erste Ableitung')
    ax3.legend()
    ax3.grid(True, alpha=0.2)

    # Plot 4: Function and second derivative
    ax4.plot(xPlot, co.f(xPlot), 'b-', linewidth=2, label='f(x) = exp(-x) * sin(x)')
    ax4.plot(xPlot, co.fDoublePrime(xPlot), 'g-', linewidth=2, label="f''(x)")
    ax4.plot(x, co.f(x), 'bo', markersize=10, label=f'f({x})')
    ax4.plot(x, co.fDoublePrime(x), 'go', markersize=10, label=f"f''({x})")
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=x, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Funktion und ihre zweite Ableitung')
    ax4.legend()
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


# ex_1()
ex_2()