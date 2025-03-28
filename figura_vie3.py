import numpy as np
import matplotlib.pyplot as plt

# Estilo de texto más grande
plt.rcParams.update({
    'font.size': 16,        # tamaño general
    'axes.labelsize': 18,   # etiquetas de ejes
    'xtick.labelsize': 14,  # ticks en x
    'ytick.labelsize': 14,  # ticks en y
    'legend.fontsize': 14,  # leyenda
    'figure.titlesize': 20  # título (si lo activas)
})

# Parámetros del sensor
b = 3
L = 2**b  # Rango del sensor: [0, 8)

# Tiempo
t = np.linspace(0, 6, 1000)

# Señal HDR asimétrica
f_in = (
    4.5 * np.sin(2 * np.pi * 0.27 * t) +
    2.0 * np.cos(2 * np.pi * 1.3 * t + 1) +
    3.7
)

# Señal del sensor saturable (clipping)
f_sat = np.clip(f_in, 0, L)

# Señal del sensor módulo (envuelta)
f_mod = np.mod(f_in, L)

# Señal reconstruida (idéntica a la original, con pequeño desplazamiento)
f_recon = f_in + 0.2

# Crear figura
plt.figure(figsize=(12, 5))  # un poco más grande también

# Zona efectiva del sensor
plt.fill_between(t, 0, L, color='skyblue', alpha=0.5, label='Rango efectivo del sensor $[0, 2^b)$')
plt.axhline(0, color='skyblue', linestyle='--', linewidth=1.5)
plt.axhline(L, color='skyblue', linestyle='--', linewidth=1.5)

# Señales
plt.plot(t, f_in, 'k-', linewidth=3, label='Señal HDR original', alpha=0.9)
plt.plot(t, f_sat, 'b--', linewidth=3, label='Sensor saturable', alpha=0.8)
plt.plot(t, f_mod, color='purple', linestyle='-.', linewidth=3, label='Sensor módulo', alpha=0.8)
plt.plot(t, f_recon, 'g:', linewidth=3, label='Reconstrucción', alpha=0.9)

# Eje Y solo con 0 y 2^b
plt.yticks([0, L], [r'$0$', r'$2^b$'])

# Etiquetas
plt.xlabel('Tiempo $t$')
plt.ylabel('Amplitud')

# Leyenda y estilo
plt.legend(loc='upper right')
plt.ylim(-3, 11)
plt.grid(True)
plt.tight_layout()
plt.show()
