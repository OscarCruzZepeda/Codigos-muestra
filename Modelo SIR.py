import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación
s = 990  # Número inicial de susceptibles
i = 10   # Número inicial de infectados
r = 0    # Número inicial de recuperados
beta = 0.3  # Tasa de infección
gamma = 0.1  # Tasa de recuperación
tmax = 160  # Tiempo máximo de simulación
dt = 1      # Paso de tiempo
simCount = 1000  # Número de simulaciones

# Almacena los resultados de cada simulación
resultados = []
ritmos_infeccion = []
ritmos_recuperacion = []

for _ in range(simCount):
    st, it, rt = s, i, r
    sList, iList, rList = [st], [it], [rt]
    ritmo_inf, ritmo_rec = [], []
    
    for _ in range(int(tmax/dt)):
        new_infected = np.random.binomial(st, beta * it / 1000)
        new_recovered = np.random.binomial(it, gamma)
        
        ritmo_inf.append(beta * st * it / 1000)
        ritmo_rec.append(gamma * it)
        
        st = max(st - new_infected, 0)
        it = max(it + new_infected - new_recovered, 0)
        rt = min(rt + new_recovered, 1000)
        
        sList.append(st)
        iList.append(it)
        rList.append(rt)
    
    resultados.append([sList, iList, rList])
    ritmos_infeccion.append(ritmo_inf)
    ritmos_recuperacion.append(ritmo_rec)

# Convertir resultados a numpy arrays para facilitar el cálculo del promedio y varianza
resultados = np.array(resultados)
ritmos_infeccion = np.array(ritmos_infeccion)
ritmos_recuperacion = np.array(ritmos_recuperacion)

# Calcular promedio y varianza para cada categoría
promedio_s = np.mean(resultados[:, 0, :], axis=0)
promedio_i = np.mean(resultados[:, 1, :], axis=0)
promedio_r = np.mean(resultados[:, 2, :], axis=0)

varianza_s = np.var(resultados[:, 0, :], axis=0)
varianza_i = np.var(resultados[:, 1, :], axis=0)
varianza_r = np.var(resultados[:, 2, :], axis=0)

# Calcular promedio de los ritmos de infección y recuperación
promedio_ritmo_infeccion = np.mean(ritmos_infeccion, axis=0)
promedio_ritmo_recuperacion = np.mean(ritmos_recuperacion, axis=0)

# Graficar resultados del modelo SIR
t = np.arange(0, tmax+1, dt)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, promedio_s, label='Susceptibles (Promedio)')
plt.plot(t, promedio_i, label='Infectados (Promedio)')
plt.plot(t, promedio_r, label='Recuperados (Promedio)')

plt.fill_between(t, promedio_s - np.sqrt(varianza_s), promedio_s + np.sqrt(varianza_s), alpha=0.2)
plt.fill_between(t, promedio_i - np.sqrt(varianza_i), promedio_i + np.sqrt(varianza_i), alpha=0.2)
plt.fill_between(t, promedio_r - np.sqrt(varianza_r), promedio_r + np.sqrt(varianza_r), alpha=0.2)

plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Simulación del Modelo SIR')
plt.grid(True)

# Graficar ritmos de infección y recuperación
plt.subplot(2, 1, 2)
t_ritmo = np.arange(0, tmax, dt)
plt.plot(t_ritmo, promedio_ritmo_infeccion, label='Ritmo de Infección')
plt.plot(t_ritmo, promedio_ritmo_recuperacion, label='Ritmo de Recuperación')

plt.xlabel('Tiempo')
plt.ylabel('Ritmo')
plt.legend()
plt.title('Ritmos de Infección y Recuperación en el Modelo SIR')
plt.grid(True)

plt.tight_layout()
plt.show()

# Mostrar promedio y varianza al final de la simulación
print("Promedio de Susceptibles:", promedio_s[-1])
print("Varianza de Susceptibles:", varianza_s[-1])

print("Promedio de Infectados:", promedio_i[-1])
print("Varianza de Infectados:", varianza_i[-1])

print("Promedio de Recuperados:", promedio_r[-1])
print("Varianza de Recuperados:", varianza_r[-1])
