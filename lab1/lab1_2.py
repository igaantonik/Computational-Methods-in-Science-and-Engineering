import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction


def generate_sequence(x0, x1, n, precision=np.single):
    sequence = np.zeros(n, dtype=precision)
    sequence[0] = x0
    sequence[1] = x1
    for k in range(2, n):
        sequence[k] = precision(2.25) * sequence[k-1] - precision(0.5) * sequence[k-2]
    return sequence


def generate_sequence_fraction(x0, x1, n):
    sequence = [Fraction(0) for _ in range(n)]
    sequence[0] = Fraction(x0)
    sequence[1] = Fraction(x1)
    for k in range(2, n):
        sequence[k] = Fraction(2.25) * sequence[k-1] - Fraction(0.5) * sequence[k-2]
    return sequence


def exact_solution(k):
    return (4**(-k))/3


x0_single = np.single(1/3)
x1_single = np.single(1/12)
x0_double = np.double(1/3)
x1_double = np.double(1/12)
x0_fraction = Fraction(1, 3)
x1_fraction = Fraction(1, 12)


sequence_single = generate_sequence(x0_single, x1_single, 10)
sequence_double = generate_sequence(x0_double, x1_double, 60, np.double)
sequence_fraction = generate_sequence_fraction(x0_fraction, x1_fraction, 225)
sequence_exact = np.array([exact_solution(k) for k in range(225)], dtype=np.double)
sequence_exact60 = np.array([exact_solution(k) for k in range(60)], dtype=np.double)
sequence_exact10 = np.array([exact_solution(k) for k in range(10)], dtype=np.double)


plt.figure()
plt.semilogy(sequence_exact, label='Exact Solution')
plt.semilogy(sequence_fraction, label='Fraction Precision', linestyle='--')
plt.semilogy(sequence_double, label='Double Precision', linestyle='dotted')
plt.semilogy(sequence_single, label='Single Precision')
plt.xlabel('k')
plt.ylabel('x')
plt.title('Ciąg w zależności od k')
plt.legend()
plt.show()

plt.figure()
relative_errors_single = np.abs((sequence_single - sequence_exact10)/sequence_exact10)
relative_errors_double = np.abs((sequence_double - sequence_exact60)/sequence_exact60)
relative_errors_fraction = np.abs((sequence_fraction-sequence_exact)/sequence_exact)
plt.semilogy(relative_errors_single, label='Single Precision')
plt.semilogy(relative_errors_double, label='Double Precision', linestyle='dotted')
plt.semilogy(relative_errors_fraction, label='Fraction Precision', linestyle='--')
plt.xlabel('k')
plt.ylabel('Relative Error')
plt.title('Błąd względny w zależności od k')
plt.legend()
plt.show()
