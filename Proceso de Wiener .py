import numpy as np
import matplotlib.pyplot as plt

# Parámetros generales
T = 100  # Tiempo total
N = 1000  # Número de pasos
dt = T/N  # Tamaño del paso de tiempo
t = np.linspace(0, T, N+1)  # Tiempo
M = 300  # Número de realizaciones

# Simulación del proceso de Wiener para una realización
W = np.zeros(N+1)
dW = np.sqrt(dt) * np.random.randn(N)
W[1:] = np.cumsum(dW)

# Gráfica de una realización del proceso de Wiener
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, W)
plt.title('Una realización del proceso de Wiener $W(t)$')
plt.xlabel('Tiempo t')
plt.ylabel('$W(t)$')
plt.grid(True)

# Realizar el proceso 300 veces y graficar todas las realizaciones
plt.subplot(2, 2, 2)
for i in range(M):
    W = np.zeros(N+1)
    dW = np.sqrt(dt) * np.random.randn(N)
    W[1:] = np.cumsum(dW)
    plt.plot(t, W, alpha=0.3)  # Graficar con transparencia

plt.title('300 realizaciones del proceso de Wiener $W(t)$')
plt.xlabel('Tiempo t')
plt.ylabel('$W(t)$')
plt.grid(True)

# Cálculo y gráfica de la varianza de W(t) conforme pasa el tiempo
var_Wt = np.zeros(N+1)

for i in range(M):
    W = np.zeros(N+1)
    dW = np.sqrt(dt) * np.random.randn(N)
    W[1:] = np.cumsum(dW)
    var_Wt += W**2

var_Wt /= M  # Promedio para obtener la varianza

plt.subplot(2, 2, 3)
plt.plot(t, var_Wt)
plt.title('Varianza de $W(t)$ en función del tiempo')
plt.xlabel('Tiempo t')
plt.ylabel('Varianza $\\mathrm{Var}(W(t))$')
plt.grid(True)

# Repetir el análisis con un tamaño de paso diferente y comparar los resultados
dt_new = 2 * dt  # Tamaño de paso diferente
N_new = int(T/dt_new)
t_new = np.linspace(0, T, N_new+1)

var_Wt_new = np.zeros(N_new+1)

for i in range(M):
    W = np.zeros(N_new+1)
    dW = np.sqrt(dt_new) * np.random.randn(N_new)
    W[1:] = np.cumsum(dW)
    var_Wt_new += W**2

var_Wt_new /= M

plt.subplot(2, 2, 4)
plt.plot(t, var_Wt, label=f'$dt = {dt}$')
plt.plot(t_new, var_Wt_new, label=f'$dt = {dt_new}$')
plt.title('Comparación de la varianza de $W(t)$ para diferentes $dt$')
plt.xlabel('Tiempo t')
plt.ylabel('Varianza $\\mathrm{Var}(W(t))$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Paso 5: Comparar con la varianza ⟨dW^2⟩ = dt^2
var_dt2 = dt**2 * np.arange(1, N+2)  # Teórica ⟨dW^2⟩ = dt^2
var_dt2_new = dt_new**2 * np.arange(1, N_new+2)

plt.figure(figsize=(12, 6))
plt.plot(t, var_Wt, label=f'Varianza simulada, $dt = {dt}$')
plt.plot(t, var_dt2, label=f'Varianza teórica $⟨dW^2⟩ = dt^2$')
plt.plot(t_new, var_Wt_new, label=f'Varianza simulada, $dt = {dt_new}$')
plt.plot(t_new, var_dt2_new, label=f'Varianza teórica $⟨dW^2⟩ = dt^2$')
plt.title('Comparación de varianzas simuladas y teóricas')
plt.xlabel('Tiempo t')
plt.ylabel('Varianza')
plt.legend()
plt.grid(True)
plt.show()
