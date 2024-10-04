import numpy as np
import matplotlib.pyplot as plt

# Parámetros del proceso de Poisson
lambda_rate = 3  # Tasa de eventos por segundo
T = 100  # Tiempo total de simulación en segundos
num_simulations = 10000  # Número de simulaciones
delta_t = 0.01  # Pequeño intervalo de tiempo

# Definición de los tiempos en los que queremos observar n(t)
t_values = np.array([0, 25, 50, 75, 100])

# Inicialización de la lista para almacenar resultados
n_t_results = np.zeros((num_simulations, len(t_values)))

# Simulación del proceso de Poisson utilizando pequeños intervalos de tiempo
for i in range(num_simulations):
    t = 0
    n_t = 0
    event_times = []
    
    # Dividimos el tiempo en pequeños intervalos y simulamos si ocurre un evento en cada uno
    while t < T:
        t += delta_t  # Incrementa el tiempo en pequeños pasos
        if np.random.rand() < (lambda_rate * delta_t):  # Probabilidad de que ocurra un evento
            n_t += 1
            event_times.append(t)
    
    # Registrar el número de eventos n(t) en los tiempos especificados
    for j, t_val in enumerate(t_values):
        n_t_results[i, j] = sum(np.array(event_times) <= t_val)

# Calcular el valor promedio y la varianza de n(t)
n_t_mean = np.mean(n_t_results, axis=0)
n_t_variance = np.var(n_t_results, axis=0)

# Graficar resultados
plt.figure(figsize=(10, 6))

# Histograma de n(t) para cada t
for i, t_val in enumerate(t_values):
    plt.hist(n_t_results[:, i], bins=30, alpha=0.5, label=f't={t_val}s')

plt.title("Distribución de eventos n(t) en diferentes tiempos")
plt.xlabel("Número de eventos n(t)")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# Graficar el valor promedio y la varianza de n(t)
plt.figure(figsize=(10, 6))

plt.plot(t_values, n_t_mean, 'o-', label='Valor promedio ⟨n(t)⟩')
plt.plot(t_values, n_t_variance, 's-', label='Varianza σ²(n(t))')

plt.title("Valor promedio y varianza de n(t) en función del tiempo")
plt.xlabel("Tiempo t (s)")
plt.ylabel("⟨n(t)⟩ y σ²(n(t))")
plt.legend()
plt.grid(True)
plt.show()


# Graficar el valor promedio y la varianza con las expresiones analíticas
plt.figure(figsize=(10, 6))

plt.plot(t_values, n_t_mean, 'o-', label='Valor promedio ⟨n(t)⟩ (Simulación)')
plt.plot(t_values, lambda_rate * t_values, '--', label='Valor promedio ⟨n(t)⟩ (Analítico)')

plt.plot(t_values, n_t_variance, 's-', label='Varianza σ²(n(t)) (Simulación)')
plt.plot(t_values, lambda_rate * t_values, '--', label='Varianza σ²(n(t)) (Analítico)')

plt.title("Comparación entre simulación y valores analíticos")
plt.xlabel("Tiempo t (s)")
plt.ylabel("⟨n(t)⟩ y σ²(n(t))")
plt.legend()
plt.grid(True)
plt.show()
