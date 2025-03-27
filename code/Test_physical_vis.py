import os
import json
import pickle
import numpy as np
import h5py
import torch
from tabulate import tabulate
from methods.dl import build_regressor

# === CONFIGURACIÓN ===
DATA_PATH = "data/TEST_test_vis_cocoa_dataset.h5"
NORM_FACTORS = {
    'Cadmium': 5.6,
    'Fermentation Level': 100,
    'Moisture': 10,
    'Polyphenols': 50
}
OUTPUT_LABELS = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]
DECIMALS = {
    "Cadmium": 2,
    "Fermentation Level": 0,
    "Moisture": 2,
    "Polyphenols": 0
}

# === UTILIDADES ===
def desnormalize(values, label):
    return values * NORM_FACTORS[label]

def redondear(valor, label):
    if isinstance(valor, np.ndarray):
        valor = valor.item()
    return round(valor, DECIMALS[label])

# === CARGAR MEJOR MODELO ML ===
def load_best_ml_model_vis():
    model_dir = "model/Machine_Learning/VIS/"
    best_r2 = -np.inf
    best_model_path = None
    for file in os.listdir(model_dir):
        if file.endswith(".pkl"):
            try:
                r2 = float(file.split('_')[-1].replace('.pkl', ''))
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_path = os.path.join(model_dir, file)
            except:
                continue
    if best_model_path:
        print(f"✅ ML modelo VIS: {os.path.basename(best_model_path)}")
        with open(best_model_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("❌ No hay modelo ML VIS")

# === CARGAR MEJOR MODELO DL ===
def load_best_dl_model_vis(num_bands, num_outputs, device):
    model_dir = "model/Deep_Learning/VIS/"
    best_r2 = -np.inf
    best_model_path = None
    for file in os.listdir(model_dir):
        if file.endswith(".pth"):
            try:
                r2 = float(file.split('_')[-1].replace('.pth', ''))
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_path = os.path.join(model_dir, file)
            except:
                continue
    if not best_model_path:
        raise FileNotFoundError("❌ No hay modelo DL VIS")

    print(f"✅ DL modelo VIS: {os.path.basename(best_model_path)}")
    model_name = os.path.basename(best_model_path).split("_")[0]
    json_path = best_model_path.replace(".pth", ".json")
    with open(json_path, "r") as f:
        hyperparams = json.load(f)
    model_dict = build_regressor(model_name, hyperparams, num_bands, num_outputs, device=device)
    model = model_dict["model"]
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === PREDICCIÓN Y TABLA ===
def run_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(DATA_PATH, 'r') as f:
        X = f['spec'][:]
        cadmium = f['cadmium'][:]
        fermentation = f['fermentation_level'][:]
        moisture = f['moisture'][:]
        polyphenols = f['polyphenols'][:]
    Y = np.stack([cadmium, fermentation, moisture, polyphenols], axis=1)

    cadmium_unicos = np.unique(cadmium)
    num_bands = X.shape[1]
    num_outputs = 4

    model_ml = load_best_ml_model_vis()
    model_dl = load_best_dl_model_vis(num_bands, num_outputs, device)

    tabla_ml = []
    tabla_dl = []

    for lote_id, cadmio_valor in enumerate(cadmium_unicos, 1):
        idx = np.where(cadmium == cadmio_valor)[0]
        X_lote = X[idx]
        Y_lote = Y[idx]
        real_vals = Y_lote[0]

        # --- ML ---
        Y_pred_ml = model_ml.predict(X_lote)
        Y_pred_ml_denorm = np.zeros_like(Y_pred_ml)
        for i, label in enumerate(OUTPUT_LABELS):
            Y_pred_ml_denorm[:, i] = desnormalize(Y_pred_ml[:, i], label)
        pred_ml_mean = np.mean(Y_pred_ml_denorm, axis=0)

        fila_ml = [lote_id]
        for i, label in enumerate(OUTPUT_LABELS):
            fila_ml.extend([
                redondear(real_vals[i], label),
                redondear(pred_ml_mean[i], label)
            ])
        tabla_ml.append(fila_ml)

        # --- DL ---
        X_tensor = torch.tensor(X_lote, dtype=torch.float32).to(device)
        with torch.no_grad():
            Y_pred_dl = model_dl(X_tensor).cpu().numpy()
        Y_pred_dl_denorm = np.zeros_like(Y_pred_dl)
        for i, label in enumerate(OUTPUT_LABELS):
            Y_pred_dl_denorm[:, i] = desnormalize(Y_pred_dl[:, i], label)
        pred_dl_mean = np.mean(Y_pred_dl_denorm, axis=0)

        fila_dl = [lote_id]
        for i, label in enumerate(OUTPUT_LABELS):
            fila_dl.extend([
                redondear(real_vals[i], label),
                redondear(pred_dl_mean[i], label)
            ])
        tabla_dl.append(fila_dl)

    # --- Mostrar ambas tablas ---
    headers = ["Lote"]
    for label in OUTPUT_LABELS:
        headers.extend([label, f"Predicción {label}"])

    print("\n📊 Comparación por Lote (VIS - Machine Learning)")
    print(tabulate(tabla_ml, headers=headers, tablefmt="grid"))

    print("\n📊 Comparación por Lote (VIS - Deep Learning)")
    print(tabulate(tabla_dl, headers=headers, tablefmt="grid"))

# === EJECUTAR ===
if __name__ == "__main__":
    run_predictions()
