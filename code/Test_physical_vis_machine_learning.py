import os
import pickle
import numpy as np
import h5py
from tabulate import tabulate

# Factores de desnormalización SOLO para predicciones
NORM_FACTORS = {
    'Cadmium': 5.6,
    'Fermentation Level': 100,
    'Moisture': 10,
    'Polyphenols': 50
}

# Propiedades en orden del modelo
OUTPUT_LABELS = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

# Función para cargar el mejor modelo de VIS según R² en el nombre del archivo
def load_best_ml_model_vis():
    model_dir = "model/Machine_Learning/VIS/"
    best_r2 = -np.inf
    best_model = None

    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            try:
                r2 = float(file.split('_')[-1].replace('.pkl', ''))
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_path = os.path.join(model_dir, file)
            except ValueError:
                continue

    if best_model_path:
        print(f"✅ Modelo VIS seleccionado: {os.path.basename(best_model_path)}")
        with open(best_model_path, "rb") as f:
            best_model = pickle.load(f)
        return best_model
    else:
        raise FileNotFoundError("❌ No se encontró modelo en VIS.")

# Desnormalización SOLO para las predicciones
def desnormalize(values, label):
    return values * NORM_FACTORS[label]

# Proceso por lote
def predict_by_lote(file_path):
    # Cargar modelo VIS
    model = load_best_ml_model_vis()

    # Cargar datos
    with h5py.File(file_path, 'r') as f:
        X = f['spec'][:]
        cadmium = f['cadmium'][:]
        fermentation = f['fermentation_level'][:]
        moisture = f['moisture'][:]
        polyphenols = f['polyphenols'][:]

    Y = np.stack([cadmium, fermentation, moisture, polyphenols], axis=1)

    # Detectar lotes únicos por cadmium
    cadmium_unicos = np.unique(cadmium)

    tabla = []
    for lote_id, cadmio_valor in enumerate(cadmium_unicos, 1):
        # Obtener índices del lote
        idx_lote = np.where(cadmium == cadmio_valor)[0]
        X_lote = X[idx_lote]
        Y_lote = Y[idx_lote]

        # Realizar predicción
        Y_pred = model.predict(X_lote)

        # Desnormalizar predicciones
        Y_pred_denorm = np.zeros_like(Y_pred)
        for i, label in enumerate(OUTPUT_LABELS):
            Y_pred_denorm[:, i] = desnormalize(Y_pred[:, i], label)

        # Valor real (sin desnormalizar, ya viene real del .h5)
        real_mean = Y_lote[0]  # Es igual para todo el lote
        pred_mean = np.mean(Y_pred_denorm, axis=0)

        DECIMALS = {
            "Cadmium": 2,
            "Fermentation Level": 0,
            "Moisture": 2,
            "Polyphenols": 0
        }

        fila = [lote_id]
        for i in range(4):
            label = OUTPUT_LABELS[i]
            real_val = real_mean[i].item() if isinstance(real_mean[i], np.ndarray) else real_mean[i]
            pred_val = pred_mean[i].item() if isinstance(pred_mean[i], np.ndarray) else pred_mean[i]
            fila.extend([
                round(real_val, DECIMALS[label]),
                round(pred_val, DECIMALS[label])
            ])

        tabla.append(fila)

    # Encabezado dinámico
    headers = ["Lote"]
    for label in OUTPUT_LABELS:
        headers.extend([label, f"Predicción {label}"])

    print("\n📊 Comparación por Lote (VIS - Machine Learning)")
    print(tabulate(tabla, headers=headers, tablefmt="grid"))

# Ejecutar
predict_by_lote("data/TEST_test_vis_cocoa_dataset.h5")
