import numpy as np
import matplotlib.pyplot as plt

# Parámetros
alpha = 0.1  # Probabilidad de error

n = 50  # Número de etapas

# Matriz de transición
P = np.array([[1-alpha, alpha], 
              [alpha, 1-alpha]])

# Elevar la matriz a la potencia 2 para obtener la probabilidad entre la etapa 0 y 2
P2 = np.linalg.matrix_power(P, 2)
prob_no_error = P2[0, 0]
print(f"Probabilidad de no ocurrir ningún error entre la etapa 0 y la etapa 2: {prob_no_error:.4f}")

# Simulación del proceso a lo largo de n etapas
state = np.array([1, 0])  # Estado inicial (X_0 = 0)
states = [state[0]]

for _ in range(n):
    state = P @ state  # Multiplicamos por la matriz de transición
    states.append(state[0])

# Graficar la probabilidad de recibir la señal correcta (0)
plt.plot(states, marker='o')
plt.title("Probabilidad de recibir la señal correcta a través de las etapas")
plt.xlabel("Etapa")
plt.ylabel("Probabilidad de recibir la señal 0")
plt.show()
