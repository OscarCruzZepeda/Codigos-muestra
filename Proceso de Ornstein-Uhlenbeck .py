import numpy as np
import matplotlib.pyplot as plt

# Parámetros
k = 1.0  # Tasa de retorno
g = 1.0  # Coeficiente de difusión
t_max = 50  # Tiempo máximo
dt = 0.05  # Incremento de tiempo
n = 1000  # Número de trayectorias

# Generar tiempos
times = np.arange(0, t_max, dt)

# Inicializar trayectorias
x = np.zeros((n, len(times)))

# Simulación
for i in range(1, len(times)):
    x[:, i] = x[:, i-1] - k * x[:, i-1] * dt + g * np.sqrt(dt) * np.random.randn(n)

# Visualización en 3D de trayectorias
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(n):
    ax.plot(times, [i]*len(times), x[i, :])

ax.set_xlabel('Tiempo')
ax.set_ylabel('Trayectorias')
ax.set_zlabel('x(t)')
plt.show()

# Comparar con la distribución estacionaria
plt.figure()
plt.hist(x[:, -1], bins=30, density=True, alpha=0.6, color='g')

# Teórica distribución estacionaria (Gaussiana)
mean_stationary = 0
var_stationary = g**2 / (2 * k)
sigma_stationary = np.sqrt(var_stationary)

# Graficar distribución teórica
x_vals = np.linspace(-3*sigma_stationary, 3*sigma_stationary, 100)
plt.plot(x_vals, (1/(sigma_stationary * np.sqrt(2 * np.pi))) * np.exp(-(x_vals - mean_stationary)**2 / (2 * sigma_stationary**2)), label='Distribución Teórica')
plt.xlabel('x(t)')
plt.ylabel('Densidad')
plt.legend()
plt.show()
