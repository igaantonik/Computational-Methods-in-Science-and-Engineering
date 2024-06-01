import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve

S = 762
I = 1
R = 0

N = S + I + R
beta = 1
gamma = 1/7

h = 0.2
t_values = np.arange(0, 14.2, h)

S_values_explicit = np.zeros(len(t_values))
I_values_explicit = np.zeros(len(t_values))
R_values_explicit = np.zeros(len(t_values))

S_values_explicit[0]= S
I_values_explicit[0]= I
R_values_explicit[0]= R

S_values_implicit = np.zeros(len(t_values))
I_values_implicit = np.zeros(len(t_values))
R_values_implicit = np.zeros(len(t_values))

S_values_implicit[0]= S
I_values_implicit[0]= I
R_values_implicit[0]= R

S_values_RK4 = np.zeros(len(t_values))
I_values_RK4 = np.zeros(len(t_values))
R_values_RK4 = np.zeros(len(t_values))

S_values_RK4[0]= S
I_values_RK4[0]= I
R_values_RK4[0]= R


# Explicit Euler Method
i = 1
for t in t_values:
    if t==0: continue
    R_old = R
    S_old = S
    I_old = I
    R = R_old + h*gamma*I_old
    S = S_old - h*beta*I_old*S_old/N
    I = I_old + h*beta*I_old*S_old/N - h*gamma*I_old
    S_values_explicit[i] = S
    I_values_explicit[i] = I
    R_values_explicit[i] = R
    i+=1

# Plot
plt.title("Rozwiązanie jawną metodą Eulera")
plt.plot(t_values, S_values_explicit, label='S(t) - Zdrowi podatni')
plt.plot(t_values, I_values_explicit, label='I(t) - Zainfekowani')
plt.plot(t_values, R_values_explicit, label='R(t) - Ozdrowiali')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()



# Implicit Euler Method
S, I, R = S_values_implicit[0], I_values_implicit[0], R_values_implicit[0]
i=1
for t in t_values:
    if t==0: continue
    S_old = S
    I_old = I
    R_old = R
    R = R_old + h*gamma*I
    S = S_old - h*beta*I*S/N
    I = I_old + h*beta*I*S/N - h*gamma*I
    S_values_implicit[i] = S
    I_values_implicit[i] = I
    R_values_implicit[i] = R
    i+=1

# Plot
plt.title("Rozwiązanie niejawną metodą Eulera")
plt.plot(t_values, S_values_implicit,label='S(t) - Zdrowi podatni')
plt.plot(t_values, I_values_implicit,label='I(t) - Zainfekowani')
plt.plot(t_values, R_values_implicit,label='R(t) - Ozdrowiali')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()



# Initial values
S, I, R = S_values_RK4[0], I_values_RK4[0], R_values_RK4[0]
# RK4 Method
def func(x):
    S, I, R = x
    eq_1 = S - S_old + h*beta*I*S/N
    eq_2 = I - I_old - h*beta*I*S/N + h*gamma*I
    eq_3 = R - R_old - h*gamma*I
    return eq_1, eq_2, eq_3

i=1
for t in t_values:
    if t==0: continue
    S_old = S
    I_old = I
    R_old = R
    R = R_old + h*gamma*I
    S = S_old - h*beta*I*S/N
    I = I_old + h*beta*I*S/N - h*gamma*I
    S_values_RK4[i] = S
    I_values_RK4[i] = I
    R_values_RK4[i] = R
    i+=1

# Plot
plt.title("Rozwiązanie metodą RK4")
plt.plot(t_values, S_values_RK4,  label='S(t) - Zdrowi podatni')
plt.plot(t_values, I_values_RK4, label='I(t) - Zainfekowani')
plt.plot(t_values, R_values_RK4, label='R(t) - Ozdrowiali')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()



# Dane rzeczywiste
real_I = np.array([1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4])
days = np.arange(0, 15)

# Estymacja parametrów modelu na podstawie danych rzeczywistych
def simulate_model(beta, gamma):
    S = 762
    I = 1
    R = 0

    I_values = [1]
    for day in range(1, 15):
        S_next = S - h * beta * S * I / N
        I_next = I + h * (beta * S * I / N - gamma * I)
        R_next = R + h * gamma * I
        
        S, I, R = S_next, I_next, R_next
        I_values.append(I)
    return np.array(I_values)

# Funkcja kosztu
def cost_function(theta):
    beta, gamma = theta
    predicted_I = simulate_model(beta, gamma)
    return np.sum((real_I - predicted_I)**2)

# Inicjalizacja współczynników
theta_initial = [1, 1/7]

# Minimalizacja funkcji kosztu
result = minimize(cost_function, theta_initial, method='Nelder-Mead')
beta_est, gamma_est = result.x

# Symulacja z estymowanymi parametrami
I_estimated = simulate_model(beta_est, gamma_est)

# Wykres porównujący dane rzeczywiste z wynikami symulacji
plt.figure(figsize=(12, 8))
plt.plot(days, real_I, 'o-', label='Rzeczywiste dane')
plt.plot(days, I_estimated, 'x-', label='Symulacja (beta={:.4f}, gamma={:.4f})'.format(beta_est, gamma_est))
plt.xlabel('Czas (dni)')
plt.ylabel('Liczba osób zakażonych')
plt.legend()
plt.title('Porównanie rzeczywistych danych z wynikami symulacji')
plt.grid()
plt.show()

# Wyświetlenie estymowanych parametrów
print(f"Estymowane wartości parametrów: beta={beta_est:.4f}, gamma={gamma_est:.4f}")
print(f"Współczynnik reprodukcji R0: {beta_est/gamma_est:.4f}")



N = 763
beta = 1
gamma = 1/7
h = 0.2
t_max = 14
S0 = 762
I0 = 1
R0 = 0

# Funkcja definiująca prawą stronę układu równań
def f(t, y):
    S, I, R = y
    dS = -beta * I * S / N
    dI = beta * I * S / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])

# Funkcja do rozwiązania układu równań metodą niejawną Eulera
def implicit_euler_method(y0, h, t_max):
    t_values = np.arange(0, t_max + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for k in range(1, len(t_values)):
        t_next = t_values[k]
        y_prev = y_values[k-1]
        
        # Definicja funkcji do znalezienia y_next przy użyciu fsolve
        def func(y_next):
            return y_next - y_prev - h * f(t_next, y_next)
        
        y_next = fsolve(func, y_prev)
        y_values[k] = y_next
    
    return t_values, y_values

# Rozwiązywanie układu równań
y0 = np.array([S0, I0, R0])
t_values, y_values = implicit_euler_method(y0, h, t_max)

# Wykres wyników
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values[:, 0], 'b-', label='S(t) - Podatni')
plt.plot(t_values, y_values[:, 1], 'r-', label='I(t) - Zainfekowani')
plt.plot(t_values, y_values[:, 2], 'g-', label='R(t) - Odzyskani')

plt.xlabel('Czas [dni]')
plt.ylabel('Liczba osób')
plt.title('Model Kermacka-McKendricka - Metoda niejawna Eulera')
plt.legend()
plt.grid(True)
plt.show()