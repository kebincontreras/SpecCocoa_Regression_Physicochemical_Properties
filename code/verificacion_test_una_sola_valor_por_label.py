import h5py
import numpy as np

# Ruta al archivo (puedes cambiar entre NIR y VIS)
file_path = "data/TEST_test_vis_cocoa_dataset.h5"

# Propiedades a analizar
properties = ["cadmium", "fermentation_level", "moisture", "polyphenols"]

# Abrir y analizar
with h5py.File(file_path, "r") as f:
    num_firmas = f['spec'].shape[0]
    print(f"🔍 Total de firmas espectrales: {num_firmas}\n")

    for prop in properties:
        valores = f[prop][:]
        unicos = np.unique(valores)

        print(f"📌 Propiedad: {prop}")
        print(f" - Número de valores únicos: {len(unicos)}")
        if len(unicos) == 1:
            print(f" - ✅ Todos los valores son iguales: {unicos[0]}")
        else:
            print(f" - ⚠️ Valores distintos. Ejemplos: {unicos[:5]}...\n")
