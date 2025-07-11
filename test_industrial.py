import os
import json
import h5py
import numpy as np
import torch
import pickle
from glob import glob
import matplotlib.pyplot as plt

# Global configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_LABELS = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]
NORM_FACTORS = {"Cadmium": 5.6, "Fermentation Level": 100, "Moisture": 10, "Polyphenols": 50}
DECIMALS = {"Cadmium": 2, "Fermentation Level": 0, "Moisture": 2, "Polyphenols": 0}

def desnormalize(values, label):
    return values * NORM_FACTORS[label]

def round_value(value, label):
    return round(value.item() if isinstance(value, np.ndarray) else value, DECIMALS[label])

def extract_r2_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {json_path} is empty or has invalid format.")
        return {}
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return {}

    if "test_metrics" in data:
        return {label: data["test_metrics"].get(label, {}).get("R2", -float('inf')) for label in OUTPUT_LABELS}
    if "test_r2" in data:
        return data["test_r2"]
    if "test" in data and "r2" in data["test"]:
        return data["test"]["r2"]
    return {}

def search_best_models_in_directory(dir_path, model_type, modality, input_size):
    best = {label: {"r2": -float('inf'), "model": None} for label in OUTPUT_LABELS}
    for file in os.listdir(dir_path):
        if not file.endswith(".json"):
            continue
        r2s = extract_r2_json(os.path.join(dir_path, file))
        for label in OUTPUT_LABELS:
            if r2s.get(label, -float('inf')) > best[label]["r2"]:
                best[label] = {"r2": r2s[label], "model": file.replace(".json", "")}
    return [
        {"Modality": modality, "Type": model_type, "Property": label,
         "Model": data["model"] or "N/A",
         "R2": round(data["r2"], 4) if data["r2"] != -float('inf') else "N/A",
         "Input": input_size}
        for label, data in best.items()
    ]

def load_input_size(modality):
    path = f"data/TEST_train_{modality.lower()}_cocoa_dataset_normalized.h5"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return "N/A"
    with h5py.File(path, 'r') as f:
        X = f['spec'][:]
    return X.shape[1]

def main():
    combinations = [("Deep_Learning", "VIS"), ("Deep_Learning", "NIR"), ("Machine_Learning", "VIS"), ("Machine_Learning", "NIR")]
    input_sizes = {modality: load_input_size(modality) for _, modality in combinations}
    final_table = []
    for type_name, modality in combinations:
        models_path = os.path.join("model", type_name, modality)
        if os.path.exists(models_path):
            final_table.extend(search_best_models_in_directory(models_path, type_name, modality, input_sizes[modality]))
        else:
            print(f"Folder not found: {models_path}")
    return final_table

def make_nir_ml_dl_comparison(final_table):
    path = "data/TEST_train_nir_cocoa_dataset_normalized.h5"
    path = "data/TEST_train_nir_cocoa_dataset_normalized.h5"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with h5py.File(path, 'r') as f:
        X = f["spec"][:]
        try:
            Y_dict = {label: f[label.lower().replace(" ", "_")][:] for label in OUTPUT_LABELS}
        except KeyError as e:
            print(f"Error loading labels from HDF5: {e}")
            return

    unique_cadmium = np.unique(Y_dict["Cadmium"])
    ml_models, dl_models = {}, {}

    for label in OUTPUT_LABELS:
        ml_row = next((row for row in final_table if row["Modality"] == "NIR" and row["Property"] == label and row["Type"] == "Machine_Learning"), None)
        dl_row = next((row for row in final_table if row["Modality"] == "NIR" and row["Property"] == label and row["Type"] == "Deep_Learning"), None)

        if ml_row and ml_row["Model"] != "N/A":
            model_path = os.path.join("model", "Machine_Learning", "NIR", ml_row["Model"].replace("_metrics", "") + ".pkl")
            try:
                with open(model_path, "rb") as f:
                    ml_models[label] = pickle.load(f)
            except:
                ml_models[label] = None

        if dl_row and dl_row["Model"] != "N/A":
            models_folder = os.path.join("model", "Deep_Learning", "NIR")
            search_pattern = os.path.join(models_folder, f"{dl_row['Model']}*.pt")
            pt_files = sorted(glob(search_pattern), key=os.path.getmtime, reverse=True)
            if pt_files:
                try:
                    model = torch.jit.load(pt_files[0], map_location=device)
                    model.eval()
                    dl_models[label] = model
                except:
                    dl_models[label] = None

    table = []
    for i, cad_val in enumerate(unique_cadmium, 1):
        idx = np.where(Y_dict["Cadmium"] == cad_val)[0]
        row = [f"Batch {i}"]
        for label in OUTPUT_LABELS:
            real = round_value(np.mean(desnormalize(Y_dict[label][idx], label)), label)
            pred_ml, pred_dl = "Error", "Error"
            if label in ml_models and ml_models[label]:
                try:
                    pred = ml_models[label].predict(X[idx])
                    if pred.ndim > 1:
                        pred = pred[:, OUTPUT_LABELS.index(label)]
                    pred_ml = round_value(np.mean(desnormalize(pred, label)), label)
                except:
                    pred_ml = "Error"
            if label in dl_models and dl_models[label]:
                try:
                    X_tensor = torch.tensor(X[idx], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        output = dl_models[label](X_tensor).cpu().numpy()
                    if output.ndim == 1:
                        pred = output
                    else:
                        pred = output[:, OUTPUT_LABELS.index(label)]
                    pred_dl = round_value(np.mean(desnormalize(pred, label)), label)
                except:
                    pred_dl = "Error"
            row.extend([real, pred_ml, pred_dl])
        table.append(row)

    # Bar visualization
    table_np = np.array(table)
    batches = table_np[:, 0]
    values = table_np[:, 1:].astype(float)
    x = np.arange(len(batches))
    bar_width = 0.2

    for j, label in enumerate(OUTPUT_LABELS):
        fig, ax = plt.subplots(figsize=(10, 5))
        real = values[:, j * 3]
        ml = values[:, j * 3 + 1]
        dl = values[:, j * 3 + 2]

        ax.bar(x - bar_width, real, width=bar_width, label="Real", alpha=0.8)
        ax.bar(x, ml, width=bar_width, label="ML", alpha=0.8)
        ax.bar(x + bar_width, dl, width=bar_width, label="DL", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(batches)
        ax.set_ylabel(label)
        ax.set_title(f"{label}: Comparison by sublot")
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    final_table = main()
    make_nir_ml_dl_comparison(final_table)
