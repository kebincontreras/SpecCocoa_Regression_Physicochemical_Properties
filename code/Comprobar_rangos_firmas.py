import os
import h5py
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def analizar_rangos():
    archivos = [f for f in os.listdir(DATA_DIR) if f.endswith("_normalized.h5")]

    if not archivos:
        print("❌ No se encontraron archivos normalizados.")
        return

    for archivo in archivos:
        path = os.path.join(DATA_DIR, archivo)
        print(f"\n📂 Archivo: {archivo}")

        try:
            with h5py.File(path, "r") as f:
                # Firmas espectrales
                if "spec" in f:
                    data = f["spec"][:]
                    min_val = np.min(data)
                    max_val = np.max(data)
                    print(f"   ➤ [spec] Rango: [{min_val:.4f}, {max_val:.4f}]")
                else:
                    print("   ⚠️ No se encontró 'spec'.")

                # Labels fisicoquímicos
                for label in ["cadmium", "fermentation_level", "moisture", "polyphenols"]:
                    if label in f:
                        data = f[label][:]
                        min_val = np.min(data)
                        max_val = np.max(data)
                        print(f"   ➤ [{label}] Rango: [{min_val:.4f}, {max_val:.4f}]")
                    else:
                        print(f"   ⚠️ No se encontró '{label}'.")

        except Exception as e:
            print(f"❌ Error al procesar {archivo}: {e}")

# Ejecutar
analizar_rangos()
