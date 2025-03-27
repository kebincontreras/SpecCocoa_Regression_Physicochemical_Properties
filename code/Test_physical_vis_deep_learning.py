import os
import json
import torch
import numpy as np
import h5py
from tabulate import tabulate
from methods.dl import build_regressor

# Factores de desnormalización SOLO para predicciones
NORM_FACTORS = {
    'Cadmium': 5.6,
    'Fermentation Level': 100,
    'Moisture': 10,
    'Polyphenols': 50
}

# Propiedades en orden del modelo
OUTPUT_LABELS = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

# Cargar mejor modelo DL por R² en el nombre
def load_best_dl_model_vis(num_bands, num_outputs, device):
    model_dir = "model/Deep_Learning/VIS/"
    best_r2 = -np.inf
    best_model = None

    for file in os.listdir(model_dir):
        if file.endswith(".pth"):
            try:
                r2 = float(file.split('_')[-1].replace('.pth', ''))
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_path = os.path.join(model_dir, file)
            except ValueError:
                continue

    if not best_model_path:
        raise FileNotFoundError("❌ No se encontró modelo DL en VIS.")

    print(f"✅ Modelo VIS DL seleccionado: {os.path.basename(best_model_path)}")

    # Obtener nombre del modelo y json asociado
    model_name = os.path.basename(best_model_path).split("_")[0]
    json_path = best_model_path.replace(".pth", ".json")

    # Cargar hiperparámetros
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

    # Construir modelo y cargar pesos
    model_dict = build_regressor(model_name, hyperparams, num_bands, num_outputs, device=device)
    model = model_dict["model"]
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

# Desnormalización SOLO para las predicciones
def desnormalize(values, label):
    return values * NORM_FACTORS[label]

# Proceso por lote - VIS DL
def predict_by_lote_dl(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar datos
    with h5py.File(file_path, 'r') as f:
        X = f['spec'][:]
        cadmium = f['cadmium'][:]
        fermentation = f['fermentation_level'][:]
        moisture = f['moisture'][:]
        polyphenols = f['polyphenols'][:]

    Y = np.stack([cadmium, fermentation, moisture, polyphenols], axis=1)

    num_bands = X.shape[1]
    num_outputs = 4

    # Cargar modelo DL
    model = load_best_dl_model_vis(num_bands, num_outputs, device)

    # Detectar lotes únicos
    cadmium_unicos = np.unique(cadmium)

    tabla = []
    for lote_id, cadmio_valor in enumerate(cadmium_unicos, 1):
        # Filtrar lote
        idx_lote = np.where(cadmium == cadmio_valor)[0]
        X_lote = X[idx_lote]
        Y_lote = Y[idx_lote]

        # Predicción con DL
        X_tensor = torch.tensor(X_lote, dtype=torch.float32).to(device)
        with torch.no_grad():
            Y_pred_tensor = model(X_tensor).cpu().numpy()

        # Desnormalizar predicciones
        Y_pred_denorm = np.zeros_like(Y_pred_tensor)
        for i, label in enumerate(OUTPUT_LABELS):
            Y_pred_denorm[:, i] = desnormalize(Y_pred_tensor[:, i], label)

        # Valor real (no desnormalizar)
        real_mean = Y_lote[0]
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

    # Encabezado tabla
    headers = ["Lote"]
    for label in OUTPUT_LABELS:
        headers.extend([label, f"Predicción {label}"])

    print("\n📊 Comparación por Lote (VIS - Deep Learning)")
    print(tabulate(tabla, headers=headers, tablefmt="grid"))

# Ejecutar
predict_by_lote_dl("data/TEST_test_vis_cocoa_dataset.h5")
