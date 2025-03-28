import numpy as np
import matplotlib.pyplot as plt

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

# Verificar rango
print(f"Rango señal HDR: min = {np.min(f_in):.2f}, max = {np.max(f_in):.2f}")

# Señal del sensor saturable (clipping)
f_sat = np.clip(f_in, 0, L)

# Señal del sensor módulo (envuelta)
f_mod = np.mod(f_in, L)

# Señal reconstruida (idéntica a la original)
f_recon = f_in

# Crear figura
plt.figure(figsize=(10, 4.5))

# Zona efectiva del sensor
plt.fill_between(t, 0, L, color='skyblue', alpha=0.25, label='Rango efectivo del sensor $[0, 2^b)$')
plt.axhline(0, color='skyblue', linestyle='--', linewidth=1.5)
plt.axhline(L, color='skyblue', linestyle='--', linewidth=1.5)

# Señal HDR original (amarillo)
plt.plot(t, f_in, color='gold', linestyle='--', linewidth=3, label='Señal HDR original')

# Señal del sensor saturable
plt.plot(t, f_sat, 'b--', linewidth=3, label='Sensor saturable')

# Señal del sensor módulo
plt.plot(t, f_mod, color='purple', linestyle='-.', linewidth=3, label='Sensor módulo')

# Reconstrucción ideal
plt.plot(t, f_recon, 'g:', linewidth=3, label='Reconstrucción ideal')

# Eje Y: solo 0 y 2^b
plt.yticks([0, L], [r'$0$', r'$2^b$'])

# Estética
plt.ylim(-3, 11)
plt.xlabel('Tiempo $t$')
plt.ylabel('Amplitud')
plt.title('Comparación: sensor saturable vs sensor módulo')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
