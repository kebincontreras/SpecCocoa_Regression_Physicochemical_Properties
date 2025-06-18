import os
import json
import h5py
import numpy as np
import torch
import pickle
from glob import glob
import matplotlib.pyplot as plt

# === CONFIGURACIÓN GLOBAL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_LABELS = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]
NORM_FACTORS = {"Cadmium": 5.6, "Fermentation Level": 100, "Moisture": 10, "Polyphenols": 50}
DECIMALS = {"Cadmium": 2, "Fermentation Level": 0, "Moisture": 2, "Polyphenols": 0}

def desnormalize(values, label):
    return values * NORM_FACTORS[label]

def redondear(valor, label):
    return round(valor.item() if isinstance(valor, np.ndarray) else valor, DECIMALS[label])

def extraer_r2_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Error al decodificar JSON: {json_path} está vacío o tiene un formato inválido.")
        return {}
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {json_path}")
        return {}

    if "test_metrics" in data:
        return {label: data["test_metrics"].get(label, {}).get("R2", -float('inf')) for label in OUTPUT_LABELS}
    if "test_r2" in data:
        return data["test_r2"]
    if "test" in data and "r2" in data["test"]:
        return data["test"]["r2"]
    return {}

def buscar_mejores_modelos_en_directorio(dir_path, tipo_modelo, modalidad, input_size):
    mejores = {label: {"r2": -float('inf'), "modelo": None} for label in OUTPUT_LABELS}
    for file in os.listdir(dir_path):
        if not file.endswith(".json"):
            continue
        r2s = extraer_r2_json(os.path.join(dir_path, file))
        for label in OUTPUT_LABELS:
            if r2s.get(label, -float('inf')) > mejores[label]["r2"]:
                mejores[label] = {"r2": r2s[label], "modelo": file.replace(".json", "")}
    return [
        {"Modalidad": modalidad, "Tipo": tipo_modelo, "Propiedad": label,
         "Modelo": data["modelo"] or "N/A",
         "R2": round(data["r2"], 4) if data["r2"] != -float('inf') else "N/A",
         "Input": input_size}
        for label, data in mejores.items()
    ]

def cargar_input_size(modalidad):  # Cambiar 'modality' a 'modalidad'
    path = f"data/TEST_train_{modalidad.lower()}_cocoa_dataset_normalized.h5"
    if not os.path.exists(path):
        print(f"❌ Archivo no encontrado: {path}")
        return "N/A"
    with h5py.File(path, 'r') as f:
        X = f['spec'][:]
    return X.shape[1]

def main():
    combinaciones = [("Deep_Learning", "VIS"), ("Deep_Learning", "NIR"), ("Machine_Learning", "VIS"), ("Machine_Learning", "NIR")]
    input_sizes = {modalidad: cargar_input_size(modalidad) for _, modalidad in combinaciones}
    tabla_final = []
    for tipo, modalidad in combinaciones:
        path_modelos = os.path.join("model", tipo, modalidad)
        if os.path.exists(path_modelos):
            tabla_final.extend(buscar_mejores_modelos_en_directorio(path_modelos, tipo, modalidad, input_sizes[modalidad]))
        else:
            print(f"❌ Carpeta no encontrada: {path_modelos}")
    return tabla_final

def hacer_comparacion_nir_ml_dl(tabla_final):
    path = "data/TEST_train_nir_cocoa_dataset_normalized.h5"
    if not os.path.exists(path):
        print(f"❌ Archivo no encontrado: {path}")
        return

    with h5py.File(path, 'r') as f:
        X = f["spec"][:]
        try:
            Y_dict = {label: f[label.lower().replace(" ", "_")][:] for label in OUTPUT_LABELS}
        except KeyError as e:
            print(f"❌ Error al cargar etiquetas del HDF5: {e}")
            return

    cadmium_unicos = np.unique(Y_dict["Cadmium"])
    modelos_ml, modelos_dl = {}, {}

    for label in OUTPUT_LABELS:
        fila_ml = next((row for row in tabla_final if row["Modalidad"] == "NIR" and row["Propiedad"] == label and row["Tipo"] == "Machine_Learning"), None)
        fila_dl = next((row for row in tabla_final if row["Modalidad"] == "NIR" and row["Propiedad"] == label and row["Tipo"] == "Deep_Learning"), None)

        if fila_ml and fila_ml["Modelo"] != "N/A":
            modelo_path = os.path.join("model", "Machine_Learning", "NIR", fila_ml["Modelo"].replace("_metrics", "") + ".pkl")
            try:
                with open(modelo_path, "rb") as f:
                    modelos_ml[label] = pickle.load(f)
            except:
                modelos_ml[label] = None

        if fila_dl and fila_dl["Modelo"] != "N/A":
            carpeta_modelos = os.path.join("model", "Deep_Learning", "NIR")
            patron_busqueda = os.path.join(carpeta_modelos, f"{fila_dl['Modelo']}*.pt")
            archivos_pt = sorted(glob(patron_busqueda), key=os.path.getmtime, reverse=True)
            if archivos_pt:
                try:
                    modelo = torch.jit.load(archivos_pt[0], map_location=device)
                    modelo.eval()
                    modelos_dl[label] = modelo
                except:
                    modelos_dl[label] = None

    tabla = []
    for i, cad_val in enumerate(cadmium_unicos, 1):
        idx = np.where(Y_dict["Cadmium"] == cad_val)[0]
        fila = [f"Lote {i}"]
        for label in OUTPUT_LABELS:
            real = redondear(np.mean(desnormalize(Y_dict[label][idx], label)), label)
            pred_ml, pred_dl = "Error", "Error"
            if label in modelos_ml and modelos_ml[label]:
                try:
                    pred = modelos_ml[label].predict(X[idx])
                    if pred.ndim > 1:
                        pred = pred[:, OUTPUT_LABELS.index(label)]
                    pred_ml = redondear(np.mean(desnormalize(pred, label)), label)
                except:
                    pred_ml = "Error"
            if label in modelos_dl and modelos_dl[label]:
                try:
                    X_tensor = torch.tensor(X[idx], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        salida = modelos_dl[label](X_tensor).cpu().numpy()
                    if salida.ndim == 1:
                        pred = salida
                    else:
                        pred = salida[:, OUTPUT_LABELS.index(label)]
                    pred_dl = redondear(np.mean(desnormalize(pred, label)), label)
                except:
                    pred_dl = "Error"
            fila.extend([real, pred_ml, pred_dl])
        tabla.append(fila)

    # === VISUALIZACIÓN DE BARRAS ===
    tabla_np = np.array(tabla)
    lotes = tabla_np[:, 0]
    valores = tabla_np[:, 1:].astype(float)
    x = np.arange(len(lotes))
    ancho_barra = 0.2

    for j, label in enumerate(OUTPUT_LABELS):
        fig, ax = plt.subplots(figsize=(10, 5))
        real = valores[:, j * 3]
        ml = valores[:, j * 3 + 1]
        dl = valores[:, j * 3 + 2]

        ax.bar(x - ancho_barra, real, width=ancho_barra, label="Real", alpha=0.8)
        ax.bar(x, ml, width=ancho_barra, label="ML", alpha=0.8)
        ax.bar(x + ancho_barra, dl, width=ancho_barra, label="DL", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(lotes)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: Comparaci\u00f3n por sublote")
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tabla_final = main()
    hacer_comparacion_nir_ml_dl(tabla_final)
