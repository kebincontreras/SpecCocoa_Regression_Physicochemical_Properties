import os
import json
import pickle
import numpy as np
import h5py
import torch
from tabulate import tabulate
from methods.dl import build_regressor

# === CONFIGURACIONES COMUNES ===
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

def load_best_dl_models(model_dir, num_bands, num_outputs, device):
    best_models = {}
    best_r2 = {label: -np.inf for label in OUTPUT_LABELS}

    for file in os.listdir(model_dir):
        if file.endswith(".json"):
            json_path = os.path.join(model_dir, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            if 'metrics_per_property' not in data:
                continue

            for label in OUTPUT_LABELS:
                r2 = data['metrics_per_property'].get(label, {}).get("R²", -np.inf)
                if r2 > best_r2[label]:
                    best_r2[label] = r2
                    model_name = file.replace(".json", "")
                    pth_path = os.path.join(model_dir, model_name + ".pth")
                    model_info = build_regressor(data['name'], data, num_bands, len(OUTPUT_LABELS), device=device)
                    model = model_info["model"]
                    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
                    model.to(device)
                    model.eval()
                    best_models[label] = model

    print(f"\n✅ Modelos DL por propiedad cargados desde {model_dir}")
    return best_models

def load_best_ml_models(model_dir):
    best_models = {}
    best_r2 = {label: -np.inf for label in OUTPUT_LABELS}

    for file in os.listdir(model_dir):
        if file.endswith("_metrics.json"):
            json_path = os.path.join(model_dir, file)
            model_base = file.replace("_metrics.json", "")
            model_path = os.path.join(model_dir, model_base + ".pkl")
            if not os.path.exists(model_path):
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)
            r2_dict = data.get("test", {}).get("r2", {})

            for label in OUTPUT_LABELS:
                r2 = r2_dict.get(label, -np.inf)
                if r2 > best_r2[label]:
                    with open(model_path, "rb") as mf:
                        model = pickle.load(mf)
                    best_models[label] = model
                    best_r2[label] = r2

    print(f"\n✅ Modelos ML por propiedad cargados desde {model_dir}")
    return best_models

def run_predictions(modality):
    print(f"\n🔎 Procesando modalidad: {modality}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = f"data/TEST_test_{modality.lower()}_cocoa_dataset.h5"
    model_dir_dl = f"model/Deep_Learning/{modality}/"
    model_dir_ml = f"model/Machine_Learning/{modality}/"

    with h5py.File(data_path, 'r') as f:
        X = f['spec'][:]
        cadmium = f['cadmium'][:]
        fermentation = f['fermentation_level'][:]
        moisture = f['moisture'][:]
        polyphenols = f['polyphenols'][:]
    Y = np.stack([cadmium, fermentation, moisture, polyphenols], axis=1)
    cadmium_unicos = np.unique(cadmium)
    num_bands = X.shape[1]

    models_dl = load_best_dl_models(model_dir_dl, num_bands, len(OUTPUT_LABELS), device)
    models_ml = load_best_ml_models(model_dir_ml)

    tabla_dl = []
    tabla_ml = []

    for lote_id, cadmio_valor in enumerate(cadmium_unicos, 1):
        idx = np.where(cadmium == cadmio_valor)[0]
        X_lote = X[idx]
        Y_lote = Y[idx]
        real_vals = Y_lote[0]

        fila_ml = [lote_id]
        fila_dl = [lote_id]
        X_tensor = torch.tensor(X_lote, dtype=torch.float32).to(device)

        for i, label in enumerate(OUTPUT_LABELS):
            real = real_vals[i]

            # --- ML ---
            model_ml = models_ml.get(label)
            if model_ml is not None:
                preds_ml = model_ml.predict(X_lote)[:, i]
                pred_ml_mean = np.mean(desnormalize(preds_ml, label))
                fila_ml.extend([redondear(real, label), redondear(pred_ml_mean, label)])
            else:
                fila_ml.extend([redondear(real, label), "N/A"])

            # --- DL ---
            model_dl = models_dl.get(label)
            if model_dl is not None:
                with torch.no_grad():
                    preds_dl = model_dl(X_tensor).cpu().numpy()[:, i]
                pred_dl_mean = np.mean(desnormalize(preds_dl, label))
                fila_dl.extend([redondear(real, label), redondear(pred_dl_mean, label)])
            else:
                fila_dl.extend([redondear(real, label), "N/A"])

        tabla_ml.append(fila_ml)
        tabla_dl.append(fila_dl)

    headers = ["Lote"]
    for label in OUTPUT_LABELS:
        headers.extend([label, f"Predicción {label}"])

    print(f"\n📊 Comparación por Lote ({modality} - Machine Learning por propiedad)")
    print(tabulate(tabla_ml, headers=headers, tablefmt="grid"))

    print(f"\n📊 Comparación por Lote ({modality} - Deep Learning por propiedad)")
    print(tabulate(tabla_dl, headers=headers, tablefmt="grid"))

# === EJECUCIÓN ===
if __name__ == "__main__":
    opcion = os.getenv("MODALIDAD", "AMBAS").strip().upper()
    if opcion == "VIS":
        run_predictions("VIS")
    elif opcion == "NIR":
        run_predictions("NIR")
    elif opcion == "AMBAS":
        run_predictions("VIS")
        run_predictions("NIR")
    else:
        print("❌ Opción no válida. Escribe VIS, NIR o AMBAS.")