import subprocess

# Ejecutar Machine Learning
print("\n================= EJECUTANDO MACHINE LEARNING =================\n")
subprocess.run(["python", "machine_learning_wb.py"], check=True)

# Ejecutar Deep Learning
print("\n================= EJECUTANDO DEEP LEARNING =================\n")
subprocess.run(["python", "deep_learning_wb.py"], check=True)


print("\n✅ Todos los experimentos han sido ejecutados correctamente.")
