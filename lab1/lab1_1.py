import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.tan(x)


def forward_difference(x, h):
    return (f(x + h) - f(x)) / h


def central_difference(x, h):
    return (f(x + h) - f(x - h)) / (2*h)


def first_derivative(x):
    return 1 + np.tan(x)**2


def second_derivative(x):
    return 2*np.tan(x) + 2*(np.tan(x)**3)


def third_derivative(x):
    return 2 + 8*(np.tan(x)**2) + 6*(np.tan(x)**4)


x0 = 1
h_values = np.power(10.0, -np.arange(0, 17))
mini = float('inf')
mini2 = float('inf')
h_min_plot = 0
h_min_plot2 = 0

truncation_errors = []
rounding_errors = []
approx_errors = []

truncation_errors2 = []
rounding_errors2 = []
approx_errors2 = []


for h in h_values:
    M = np.abs(second_derivative(x0))
    approx_value = forward_difference(x0, h)
    t_value = M * h / 2
    r_value = 2*np.finfo(float).eps/h
    a_value = np.abs(first_derivative(x0) - approx_value)
    truncation_errors.append(t_value)
    rounding_errors.append(r_value)
    approx_errors.append(a_value)
    if a_value < mini:
        mini = a_value
        h_min_plot = h

    M2 = np.abs(third_derivative(x0))
    approx_value2 = central_difference(x0, h)
    t_value2 = M2*(h**2)/6
    r_value2 = np.finfo(float).eps/h
    a_value2 = np.abs(first_derivative(x0) - approx_value2)
    truncation_errors2.append(t_value2)
    rounding_errors2.append(r_value2)
    approx_errors2.append(a_value2)
    if a_value2 < mini2:
        mini2 = a_value2
        h_min_plot2 = h


h_min = 2 * np.sqrt(np.finfo(float).eps / np.abs(second_derivative(x0)))
print("h_min using formula: ", h_min, ", h_min on plot: ", h_min_plot)
print("Difference of h_min in forward difference method: ", np.abs(h_min - h_min_plot))


h_min2 = np.cbrt(3*np.finfo(float).eps / np.abs(third_derivative(x0)))
print("h_min using formula: ", h_min2, ", h_min on plot: ", h_min_plot2)
print("Difference of h_min in central difference method: ", np.abs(h_min2 - h_min_plot2))

plt.figure()
plt.loglog(h_values, truncation_errors, label='TruncationError', marker='o')
plt.loglog(h_values, rounding_errors, label='Rounding Error', marker='o')
plt.loglog(h_values, approx_errors, label='Approximation Error', marker='o')
plt.xlabel('h')
plt.ylabel('Error')
plt.title('Errors for forward difference method of tan(x)')
plt.legend()
plt.show()


plt.figure()
plt.loglog(h_values, truncation_errors2, label='TruncationError', marker='o')
plt.loglog(h_values, rounding_errors2, label='Rounding Error', marker='o')
plt.loglog(h_values, approx_errors2, label='Approximation Error', marker='o')
plt.xlabel('h')
plt.ylabel('Error')
plt.title('Errors for central difference method of tan(x)')
plt.legend()
plt.show()







