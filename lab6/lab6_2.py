import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

def f(x):
    return 4 / (1 + x**2)

def gauss_legendre_integration(n):
    # Węzły i wagi Legendre'a dla n punktów
    x, w = roots_legendre(n)
    
    # Skalowanie węzłów do przedziału [0, 1]
    x_scaled = 0.5 * (x + 1)
    w_scaled = 0.5 * w
    
    integral = np.sum(w_scaled * f(x_scaled))
    return integral


exact_integral = np.pi
errors_gauss_legendre = []
m_values = np.arange(0, 10)

for m in m_values:
    n = 2**m+1
    integral_approx = gauss_legendre_integration(n)
    error = np.abs((exact_integral - integral_approx) / exact_integral)
    errors_gauss_legendre.append(error)

plt.figure(figsize=(10, 6))
plt.plot((2**m_values+1) + 1, errors_gauss_legendre, color='purple', label='Gauss-Legendre Integration')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Number of evaluations')
plt.ylabel('Relative error')
plt.title('Convergence of Numerical Integration Methods')
plt.legend()
plt.grid(True)
plt.show()