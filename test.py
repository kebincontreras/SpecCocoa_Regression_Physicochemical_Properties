import warnings
warnings.filterwarnings("ignore")
import os
import json
import h5py
import numpy as np
import torch
import pickle
from glob import glob

OUTPUT_LABELS = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]
NORM_FACTORS = {"Cadmium": 5.6, "Fermentation Level": 100, "Moisture": 10, "Polyphenols": 50}

# Configuración para datos industriales y de validación
INDUSTRIAL_MODALITIES = [
    {"name": "NIR", "h5": "data/test_nir_cocoa_dataset_normalized.h5"},
    {"name": "VIS", "h5": "data/test_regions_vis_cocoa_dataset_normalized.h5"}
]

VALIDATION_MODALITIES = [
    {"name": "NIR", "h5": "data/test_val_nir_cocoa_dataset_normalized.h5"},
    {"name": "VIS", "h5": "data/test_val_vis_cocoa_dataset_normalized.h5"}
]

def desnormalize(values, label):
    return values * NORM_FACTORS[label]

def round_value(value):
    v = value.item() if isinstance(value, np.ndarray) else value
    return round(v, 2)

def find_best_dl_models(model_dir, device):
    dl_models = {}
    dl_model_files = {}
    best_r2 = {label: -float('inf') for label in OUTPUT_LABELS}
    best_model_file = {label: None for label in OUTPUT_LABELS}
    
    for file in os.listdir(model_dir):
        if file.endswith('.json'):
            json_path = os.path.join(model_dir, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            if 'test_metrics' in data:
                for label in OUTPUT_LABELS:
                    r2 = data['test_metrics'].get(label, {}).get('R2', -float('inf'))
                    if r2 is not None and r2 > best_r2[label]:
                        best_r2[label] = r2
                        best_model_file[label] = file
                        
    for label in OUTPUT_LABELS:
        if best_model_file[label]:
            base = best_model_file[label].replace('.json', '')
            pt_pattern = os.path.join(model_dir, f"{base}*.pt")
            pt_files = sorted(glob(pt_pattern), key=os.path.getmtime, reverse=True)
            if pt_files:
                try:
                    model = torch.jit.load(pt_files[0], map_location=device)
                    model.eval()
                    dl_models[label] = model
                    dl_model_files[label] = os.path.basename(pt_files[0])
                except Exception as e:
                    pass
            else:
                dl_model_files[label] = None
        else:
            dl_model_files[label] = None
    return dl_models, dl_model_files

def find_best_ml_models(ml_dir):
    ml_models = {}
    ml_model_files = {}
    
    for label in OUTPUT_LABELS:
        best_r2 = -float('inf')
        best_model_json = None
        best_model_pkl = None
        
        for file in os.listdir(ml_dir):
            if file.endswith('.json'):
                json_path = os.path.join(ml_dir, file)
                try:
                    with open(json_path, 'r') as fjson:
                        data = json.load(fjson)
                    r2 = None
                    if 'test_metrics' in data:
                        r2 = data['test_metrics'].get(label, {}).get('R2', None)
                    elif 'test' in data and 'r2' in data['test']:
                        r2 = data['test']['r2'].get(label, None)
                    if r2 is not None and r2 > best_r2:
                        best_r2 = r2
                        best_model_json = file
                except Exception as e:
                    print(f"Error leyendo {json_path}: {e}")
                    
        if best_model_json:
            pkl_candidate = os.path.join(ml_dir, best_model_json.replace('.json', '.pkl'))
            pkl_candidate_alt = None
            if '_metrics' in pkl_candidate:
                pkl_candidate_alt = pkl_candidate.replace('_metrics', '')
            if os.path.exists(pkl_candidate):
                best_model_pkl = pkl_candidate
            elif pkl_candidate_alt and os.path.exists(pkl_candidate_alt):
                best_model_pkl = pkl_candidate_alt
                
        if best_model_pkl:
            try:
                with open(best_model_pkl, "rb") as f:
                    ml_models[label] = pickle.load(f)
                ml_model_files[label] = os.path.basename(best_model_pkl)
            except Exception as e:
                print(f"Error cargando modelo ML para {label}: {e}")
                ml_model_files[label] = None
        else:
            esperado = best_model_json.replace('.json','.pkl') if best_model_json else 'N/A'
            if best_model_json and '_metrics' in esperado:
                esperado += ' o ' + esperado.replace('_metrics','')
            print(f"No se encontró modelo ML para {label} (esperado: {esperado})")
            ml_model_files[label] = None
    return ml_models, ml_model_files

def find_global_best_models(modalities, device):
    """Busca los mejores modelos globales (un modelo para todas las propiedades)"""
    global_ml_models = {}
    global_dl_models = {}
    global_ml_model_files = {}
    global_dl_model_files = {}
    
    for mod in ["VIS", "NIR"]:
        # ML: buscar el modelo con mayor suma de R2 en todas las propiedades
        ml_dir = os.path.join("model", "Machine_Learning", mod)
        best_sum_r2 = -float('inf')
        best_model_json = None
        best_model_pkl = None
        
        for file in os.listdir(ml_dir):
            if file.endswith('.json'):
                try:
                    with open(os.path.join(ml_dir, file), 'r') as fjson:
                        d = json.load(fjson)
                    r2s = []
                    if 'test_metrics' in d:
                        for label in OUTPUT_LABELS:
                            r2 = d['test_metrics'].get(label, {}).get('R2', None)
                            if r2 is not None:
                                r2s.append(r2)
                    elif 'test' in d and 'r2' in d['test']:
                        for label in OUTPUT_LABELS:
                            r2 = d['test']['r2'].get(label, None)
                            if r2 is not None:
                                r2s.append(r2)
                    if len(r2s) == len(OUTPUT_LABELS):
                        sum_r2 = sum(r2s)
                        if sum_r2 > best_sum_r2:
                            best_sum_r2 = sum_r2
                            best_model_json = file
                except Exception:
                    continue
                    
        if best_model_json:
            pkl_candidate = os.path.join(ml_dir, best_model_json.replace('.json', '.pkl'))
            pkl_candidate_alt = None
            if '_metrics' in pkl_candidate:
                pkl_candidate_alt = pkl_candidate.replace('_metrics', '')
            if os.path.exists(pkl_candidate):
                best_model_pkl = pkl_candidate
            elif pkl_candidate_alt and os.path.exists(pkl_candidate_alt):
                best_model_pkl = pkl_candidate_alt
            else:
                best_model_pkl = None
            if best_model_pkl:
                with open(best_model_pkl, "rb") as f:
                    global_ml_models[mod] = pickle.load(f)
                global_ml_model_files[mod] = os.path.basename(best_model_pkl)
            else:
                global_ml_models[mod] = None
                global_ml_model_files[mod] = None
        else:
            global_ml_models[mod] = None
            global_ml_model_files[mod] = None
            
        # DL: buscar el modelo con mayor suma de R2 en todas las propiedades
        dl_dir = os.path.join("model", "Deep_Learning", mod)
        best_sum_r2 = -float('inf')
        best_model_json = None
        
        for file in os.listdir(dl_dir):
            if file.endswith('.json'):
                try:
                    with open(os.path.join(dl_dir, file), 'r') as fjson:
                        d = json.load(fjson)
                    r2s = []
                    if 'test_metrics' in d:
                        for label in OUTPUT_LABELS:
                            r2 = d['test_metrics'].get(label, {}).get('R2', None)
                            if r2 is not None:
                                r2s.append(r2)
                    if len(r2s) == len(OUTPUT_LABELS):
                        sum_r2 = sum(r2s)
                        if sum_r2 > best_sum_r2:
                            best_sum_r2 = sum_r2
                            best_model_json = file
                except Exception:
                    continue
                    
        if best_model_json:
            base = best_model_json.replace('.json', '')
            pt_pattern = os.path.join(dl_dir, f"{base}*.pt")
            pt_files = sorted(glob(pt_pattern), key=os.path.getmtime, reverse=True)
            if pt_files:
                try:
                    model = torch.jit.load(pt_files[0], map_location=device)
                    model.eval()
                    global_dl_models[mod] = model
                    global_dl_model_files[mod] = os.path.basename(pt_files[0])
                except Exception:
                    global_dl_models[mod] = None
                    global_dl_model_files[mod] = None
            else:
                global_dl_models[mod] = None
                global_dl_model_files[mod] = None
        else:
            global_dl_models[mod] = None
            global_dl_model_files[mod] = None
            
    return global_ml_models, global_dl_models, global_ml_model_files, global_dl_model_files

def get_modality_data(modality, device):
    h5_path = modality["h5"]
    if not os.path.exists(h5_path):
        print(f"Archivo no encontrado: {h5_path}")
        return None
        
    with h5py.File(h5_path, 'r') as f:
        X = f["spec"][:]
        try:
            Y_dict = {label: f[label.lower().replace(" ", "_")][:] for label in OUTPUT_LABELS}
        except KeyError as e:
            print(f"Error loading labels from HDF5: {e}")
            return None
            
    unique_cadmium = np.unique(Y_dict["Cadmium"])
    return {"X": X, "Y_dict": Y_dict, "unique_cadmium": unique_cadmium}

def print_combined_table(industrial_data, validation_data, global_ml_models, global_dl_models):
    """Imprimir tabla combinada con los 3 primeros batches industriales + batch 4 de validación, igual que el formato de validacion copy 2."""
    print("\nTABLA COMBINADA: Batches Industriales (Putumayo, Huila, Peru) + Batch 4 Validación (Santander)")
    print("="*100)
    header = f"{'Propiedad':20s} | {'No. real':8s} | VIS ML  | NIR ML  | VIS DL  | NIR DL  |"
    print(header)
    print("-" * len(header))
    region_names = ['Putumayo', 'Huila', 'Peru', 'Santander']
    # ===== PROCESAR PRIMEROS 3 BATCHES INDUSTRIALES =====
    all_cadmium = []
    for mod in industrial_data:
        if industrial_data[mod] is not None:
            all_cadmium.extend(industrial_data[mod]["unique_cadmium"])
    union_cadmium = np.unique(all_cadmium)[:3]
    for i, cad_val in enumerate(union_cadmium):
        print(f"Batch {i+1} - {region_names[i]}")
        idxs = {}
        for mod in industrial_data:
            if industrial_data[mod] is not None:
                idxs[mod] = np.where(industrial_data[mod]["Y_dict"]["Cadmium"] == cad_val)[0]
        for label in OUTPUT_LABELS:
            row = f"{label:20s} | "
            # Valor real (VIS si existe, si no NIR)
            real = None
            for mod in ["VIS", "NIR"]:
                if mod in industrial_data and industrial_data[mod] is not None and len(idxs[mod]) > 0:
                    real = round_value(np.mean(desnormalize(industrial_data[mod]["Y_dict"][label][idxs[mod]], label)))
                    break
            if real is not None:
                row += f"{real:8.2f} | "
            else:
                row += f"   N/A   | "
            # ML predicciones
            for mod in ["VIS", "NIR"]:
                if mod in industrial_data and industrial_data[mod] is not None and mod in global_ml_models and global_ml_models[mod] and len(idxs[mod]) > 0:
                    try:
                        pred_ml = global_ml_models[mod].predict(industrial_data[mod]["X"][idxs[mod]])
                        if pred_ml.ndim > 1:
                            pred_ml = pred_ml[:, OUTPUT_LABELS.index(label)]
                        pred_ml = desnormalize(pred_ml, label)
                        # Normalizar predicción de Fermentation Level
                        if label == "Fermentation Level":
                            pred_ml = np.clip(pred_ml, None, 100)
                        row += f"{round_value(np.mean(pred_ml)):8.2f} | "
                    except Exception:
                        row += f"   N/A   | "
                else:
                    row += f"   N/A   | "
            # DL predicciones
            for mod in ["VIS", "NIR"]:
                if mod in industrial_data and industrial_data[mod] is not None and mod in global_dl_models and global_dl_models[mod] and len(idxs[mod]) > 0:
                    try:
                        X_batch = industrial_data[mod]["X"][idxs[mod]]
                        X_tensor = torch.tensor(X_batch, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X_batch, dtype=torch.float32)
                        with torch.no_grad():
                            output = global_dl_models[mod](X_tensor)
                            if output.is_cuda:
                                output = output.cpu()
                            output = output.numpy()
                        if output.ndim == 1:
                            pred_dl = output
                        else:
                            pred_dl = output[:, OUTPUT_LABELS.index(label)]
                        pred_dl = desnormalize(pred_dl, label)
                        # Normalizar predicción de Fermentation Level
                        if label == "Fermentation Level":
                            pred_dl = np.clip(pred_dl, None, 100)
                        row += f"{round_value(np.mean(pred_dl)):8.2f} | "
                        del X_tensor, output
                    except Exception:
                        row += f"   N/A   | "
                else:
                    row += f"   N/A   | "
            print(row)
        print()
    # ===== PROCESAR BATCH 4 DE VALIDACIÓN (SANTANDER) =====
    print(f"Batch 4 - {region_names[3]} (Fermentación 0.96)")
    for label in OUTPUT_LABELS:
        row = f"{label:20s} | "
        # Valor real (VIS si existe, si no NIR)
        real = None
        for mod in ["VIS", "NIR"]:
            if mod in validation_data and validation_data[mod] is not None:
                fermentation_levels = validation_data[mod]["Y_dict"]["Fermentation Level"]
                idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
                if len(idxs) > 0:
                    real = round_value(np.mean(desnormalize(validation_data[mod]["Y_dict"][label][idxs], label)))
                    break
        if real is not None:
            row += f"{real:8.2f} | "
        else:
            row += f"   N/A   | "
        # ML predicciones
        for mod in ["VIS", "NIR"]:
            if mod in validation_data and validation_data[mod] is not None and mod in global_ml_models and global_ml_models[mod]:
                fermentation_levels = validation_data[mod]["Y_dict"]["Fermentation Level"]
                idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
                if len(idxs) > 0:
                    try:
                        pred_ml = global_ml_models[mod].predict(validation_data[mod]["X"][idxs])
                        if pred_ml.ndim > 1:
                            pred_ml = pred_ml[:, OUTPUT_LABELS.index(label)]
                        pred_ml = desnormalize(pred_ml, label)
                        # Normalizar predicción de Fermentation Level
                        if label == "Fermentation Level":
                            pred_ml = np.clip(pred_ml, None, 100)
                        row += f"{round_value(np.mean(pred_ml)):8.2f} | "
                    except Exception:
                        row += f"   N/A   | "
                else:
                    row += f"   N/A   | "
            else:
                row += f"   N/A   | "
        # DL predicciones
        for mod in ["VIS", "NIR"]:
            if mod in validation_data and validation_data[mod] is not None and mod in global_dl_models and global_dl_models[mod]:
                fermentation_levels = validation_data[mod]["Y_dict"]["Fermentation Level"]
                idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
                if len(idxs) > 0:
                    try:
                        X_batch = validation_data[mod]["X"][idxs]
                        X_tensor = torch.tensor(X_batch, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X_batch, dtype=torch.float32)
                        with torch.no_grad():
                            output = global_dl_models[mod](X_tensor)
                            if output.is_cuda:
                                output = output.cpu()
                            output = output.numpy()
                        if output.ndim == 1:
                            pred_dl = output
                        else:
                            pred_dl = output[:, OUTPUT_LABELS.index(label)]
                        pred_dl = desnormalize(pred_dl, label)
                        # Normalizar predicción de Fermentation Level
                        if label == "Fermentation Level":
                            pred_dl = np.clip(pred_dl, None, 100)
                        row += f"{round_value(np.mean(pred_dl)):8.2f} | "
                        del X_tensor, output
                    except Exception:
                        row += f"   N/A   | "
                else:
                    row += f"   N/A   | "
            else:
                row += f"   N/A   | "
        print(row)
    print()
def create_combined_plots(industrial_data, validation_data, global_ml_models, global_dl_models):
    """Generate only the combined graphic for the 4 batches (3 industrial + 1 validation), with all labels in English, no title, and legend below the entire figure."""
    import matplotlib.pyplot as plt
    import matplotlib
    import warnings as _warnings
    _warnings.filterwarnings("ignore", message="No artists with labels found to put in legend.")
    matplotlib.use('Agg')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    pastel_ml = '#a3c1da'
    pastel_dl = '#b7e0c7'
    pastel_ml_nir = '#7ea7d8'
    pastel_dl_nir = '#6fd1a7'

    props = OUTPUT_LABELS
    region_names = ['Putumayo', 'Huila', 'Peru', 'Santander']
    prop_labels_en = ['Cadmium', 'Fermentation Level', 'Moisture', 'Polyphenols']
    all_cadmium = []
    for mod in industrial_data:
        if industrial_data[mod] is not None:
            all_cadmium.extend(industrial_data[mod]["unique_cadmium"])
    industrial_batches = np.unique(all_cadmium)[:3]
    total_batches = 4

    fig, axes = plt.subplots(total_batches, len(props), figsize=(5*len(props), 4.5*total_batches), sharex=False)
    if total_batches == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(props) == 1:
        axes = np.expand_dims(axes, axis=1)

    # Prepare legend handles (for all subplots)
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=pastel_ml, label='VIS-ML'),
        mpatches.Patch(facecolor=pastel_dl, label='VIS-DL'),
        mpatches.Patch(facecolor=pastel_ml_nir, label='NIR-ML', hatch='///'),
        mpatches.Patch(facecolor=pastel_dl_nir, label='NIR-DL', hatch='xx'),
        plt.Line2D([0], [0], color='#2ecc40', linestyle='--', linewidth=2, label='Reference value')
    ]

    # Process the 3 industrial batches
    for batch_idx, cad_val in enumerate(industrial_batches):
        region = region_names[batch_idx]
        for prop_idx, prop in enumerate(props):
            ax = axes[batch_idx, prop_idx]
            box_data = []
            box_colors = [pastel_ml, pastel_dl, pastel_ml_nir, pastel_dl_nir]
            box_hatches = [None, None, '///', 'xx']
            xtick_labels = ['VIS-ML', 'VIS-DL', 'NIR-ML', 'NIR-DL']
            # VIS-ML
            vals = [np.nan]
            if 'VIS' in industrial_data and industrial_data['VIS'] is not None and 'VIS' in global_ml_models and global_ml_models['VIS']:
                idxs = np.where(industrial_data['VIS']['Y_dict']['Cadmium'] == cad_val)[0]
                if len(idxs) > 0:
                    pred_ml = global_ml_models['VIS'].predict(industrial_data['VIS']['X'][idxs])
                    if pred_ml.ndim > 1:
                        pred_ml = pred_ml[:, OUTPUT_LABELS.index(prop)]
                    pred_ml = desnormalize(pred_ml, prop).flatten()
                    if prop == "Fermentation Level":
                        pred_ml = np.clip(pred_ml, None, 100)
                    vals = pred_ml
            box_data.append(vals)
            # VIS-DL
            vals = [np.nan]
            if 'VIS' in industrial_data and industrial_data['VIS'] is not None and 'VIS' in global_dl_models and global_dl_models['VIS']:
                idxs = np.where(industrial_data['VIS']['Y_dict']['Cadmium'] == cad_val)[0]
                if len(idxs) > 0:
                    batch_size = 32
                    preds = []
                    for start in range(0, len(idxs), batch_size):
                        end = start + batch_size
                        X_batch = industrial_data['VIS']['X'][idxs][start:end]
                        X_tensor = torch.tensor(X_batch, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X_batch, dtype=torch.float32)
                        with torch.no_grad():
                            output = global_dl_models['VIS'](X_tensor)
                            if output.is_cuda:
                                output = output.cpu()
                            output = output.numpy()
                        if output.ndim == 1:
                            pred = output
                        else:
                            pred = output[:, OUTPUT_LABELS.index(prop)]
                        pred = desnormalize(pred, prop)
                        if prop == "Fermentation Level":
                            pred = np.clip(pred, None, 100)
                        preds.append(pred)
                        del X_tensor, output
                    if len(preds) > 0:
                        preds = np.concatenate(preds)
                        vals = preds
            box_data.append(vals)
            # NIR-ML
            vals = [np.nan]
            if 'NIR' in industrial_data and industrial_data['NIR'] is not None and 'NIR' in global_ml_models and global_ml_models['NIR']:
                idxs = np.where(industrial_data['NIR']['Y_dict']['Cadmium'] == cad_val)[0]
                if len(idxs) > 0:
                    pred_ml = global_ml_models['NIR'].predict(industrial_data['NIR']['X'][idxs])
                    if pred_ml.ndim > 1:
                        pred_ml = pred_ml[:, OUTPUT_LABELS.index(prop)]
                    pred_ml = desnormalize(pred_ml, prop).flatten()
                    if prop == "Fermentation Level":
                        pred_ml = np.clip(pred_ml, None, 100)
                    vals = pred_ml
            box_data.append(vals)
            # NIR-DL
            vals = [np.nan]
            if 'NIR' in industrial_data and industrial_data['NIR'] is not None and 'NIR' in global_dl_models and global_dl_models['NIR']:
                idxs = np.where(industrial_data['NIR']['Y_dict']['Cadmium'] == cad_val)[0]
                if len(idxs) > 0:
                    batch_size = 32
                    preds = []
                    for start in range(0, len(idxs), batch_size):
                        end = start + batch_size
                        X_batch = industrial_data['NIR']['X'][idxs][start:end]
                        X_tensor = torch.tensor(X_batch, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X_batch, dtype=torch.float32)
                        with torch.no_grad():
                            output = global_dl_models['NIR'](X_tensor)
                            if output.is_cuda:
                                output = output.cpu()
                            output = output.numpy()
                        if output.ndim == 1:
                            pred = output
                        else:
                            pred = output[:, OUTPUT_LABELS.index(prop)]
                        pred = desnormalize(pred, prop)
                        if prop == "Fermentation Level":
                            pred = np.clip(pred, None, 100)
                        preds.append(pred)
                        del X_tensor, output
                    if len(preds) > 0:
                        preds = np.concatenate(preds)
                        vals = preds
            box_data.append(vals)
            # Real value
            real_val = np.nan
            for mod in ['VIS', 'NIR']:
                if mod in industrial_data and industrial_data[mod] is not None:
                    idxs = np.where(industrial_data[mod]['Y_dict']['Cadmium'] == cad_val)[0]
                    if len(idxs) > 0:
                        real_val = np.mean(desnormalize(industrial_data[mod]['Y_dict'][prop][idxs], prop))
                        break
            positions = [0.7, 1.1, 1.9, 2.3]
            box = ax.boxplot(
                box_data,
                patch_artist=True,
                widths=0.22,
                positions=positions,
                showfliers=False,
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=1.5),
                flierprops=dict(markeredgewidth=1.5)
            )
            for i, (patch, color) in enumerate(zip(box['boxes'], box_colors)):
                patch.set_facecolor(color)
                if box_hatches[i]:
                    patch.set_hatch(box_hatches[i])
            if not np.isnan(real_val):
                ax.axhline(real_val, color='#2ecc40', linestyle='--', linewidth=2, label='Reference value')
            y_min = ax.get_ylim()[0]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.04
            for i, label in enumerate(xtick_labels):
                ax.text(positions[i], y_min, label, ha='center', va='top', fontsize=12, fontweight='bold', color='black', rotation=0)
            # Remove fermentation level text
            # y_min2 = ax.get_ylim()[0]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.12
            # ax.text(0.9, y_min2, 'VIS', ha='center', va='top', fontsize=13, color='#333333', fontweight='bold', rotation=0)
            # ax.text(2.1, y_min2, 'NIR', ha='center', va='top', fontsize=13, color='#333333', fontweight='bold', rotation=0)
            ax.set_xticks([])
            ax.tick_params(axis='both', which='major', labelsize=13)
            if batch_idx == 0:
                ax.set_title(prop_labels_en[prop_idx])
            if prop_idx == 0:
                ax.set_ylabel(f'{region}\nValue')
    # Process batch 4 validation
    batch_idx = 3
    region = region_names[batch_idx]
    for prop_idx, prop in enumerate(props):
        ax = axes[batch_idx, prop_idx]
        box_data = []
        box_colors = [pastel_ml, pastel_dl, pastel_ml_nir, pastel_dl_nir]
        box_hatches = [None, None, '///', 'xx']
        xtick_labels = ['VIS-ML', 'VIS-DL', 'NIR-ML', 'NIR-DL']
        # VIS-ML validation
        vals = [np.nan]
        if 'VIS' in validation_data and validation_data['VIS'] is not None and 'VIS' in global_ml_models and global_ml_models['VIS']:
            fermentation_levels = validation_data['VIS']['Y_dict']['Fermentation Level']
            idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
            if len(idxs) > 0:
                pred_ml = global_ml_models['VIS'].predict(validation_data['VIS']['X'][idxs])
                if pred_ml.ndim > 1:
                    pred_ml = pred_ml[:, OUTPUT_LABELS.index(prop)]
                pred_ml = desnormalize(pred_ml, prop).flatten()
                if prop == "Fermentation Level":
                    pred_ml = np.clip(pred_ml, None, 100)
                vals = pred_ml
        box_data.append(vals)
        # VIS-DL validation
        vals = [np.nan]
        if 'VIS' in validation_data and validation_data['VIS'] is not None and 'VIS' in global_dl_models and global_dl_models['VIS']:
            fermentation_levels = validation_data['VIS']['Y_dict']['Fermentation Level']
            idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
            if len(idxs) > 0:
                batch_size = 32
                preds = []
                for start in range(0, len(idxs), batch_size):
                    end = start + batch_size
                    X_batch = validation_data['VIS']['X'][idxs][start:end]
                    X_tensor = torch.tensor(X_batch, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X_batch, dtype=torch.float32)
                    with torch.no_grad():
                        output = global_dl_models['VIS'](X_tensor)
                        if output.is_cuda:
                            output = output.cpu()
                        output = output.numpy()
                    if output.ndim == 1:
                        pred = output
                    else:
                        pred = output[:, OUTPUT_LABELS.index(prop)]
                    pred = desnormalize(pred, prop)
                    if prop == "Fermentation Level":
                        pred = np.clip(pred, None, 100)
                    preds.append(pred)
                    del X_tensor, output
                if len(preds) > 0:
                    preds = np.concatenate(preds)
                    vals = preds
        box_data.append(vals)
        # NIR-ML validation
        vals = [np.nan]
        if 'NIR' in validation_data and validation_data['NIR'] is not None and 'NIR' in global_ml_models and global_ml_models['NIR']:
            fermentation_levels = validation_data['NIR']['Y_dict']['Fermentation Level']
            idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
            if len(idxs) > 0:
                pred_ml = global_ml_models['NIR'].predict(validation_data['NIR']['X'][idxs])
                if pred_ml.ndim > 1:
                    pred_ml = pred_ml[:, OUTPUT_LABELS.index(prop)]
                pred_ml = desnormalize(pred_ml, prop).flatten()
                if prop == "Fermentation Level":
                    pred_ml = np.clip(pred_ml, None, 100)
                vals = pred_ml
        box_data.append(vals)
        # NIR-DL validation
        vals = [np.nan]
        if 'NIR' in validation_data and validation_data['NIR'] is not None and 'NIR' in global_dl_models and global_dl_models['NIR']:
            fermentation_levels = validation_data['NIR']['Y_dict']['Fermentation Level']
            idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
            if len(idxs) > 0:
                batch_size = 32
                preds = []
                for start in range(0, len(idxs), batch_size):
                    end = start + batch_size
                    X_batch = validation_data['NIR']['X'][idxs][start:end]
                    X_tensor = torch.tensor(X_batch, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X_batch, dtype=torch.float32)
                    with torch.no_grad():
                        output = global_dl_models['NIR'](X_tensor)
                        if output.is_cuda:
                            output = output.cpu()
                        output = output.numpy()
                    if output.ndim == 1:
                        pred = output
                    else:
                        pred = output[:, OUTPUT_LABELS.index(prop)]
                    pred = desnormalize(pred, prop)
                    if prop == "Fermentation Level":
                        pred = np.clip(pred, None, 100)
                    preds.append(pred)
                    del X_tensor, output
                if len(preds) > 0:
                    preds = np.concatenate(preds)
                    vals = preds
        box_data.append(vals)
        # Real value
        real_val = np.nan
        for mod in ['VIS', 'NIR']:
            if mod in validation_data and validation_data[mod] is not None:
                fermentation_levels = validation_data[mod]['Y_dict']['Fermentation Level']
                idxs = np.where(np.abs(fermentation_levels - 0.96) < 0.001)[0]
                if len(idxs) > 0:
                    real_val = np.mean(desnormalize(validation_data[mod]['Y_dict'][prop][idxs], prop))
                    break
        positions = [0.7, 1.1, 1.9, 2.3]
        box = ax.boxplot(
            box_data,
            patch_artist=True,
            widths=0.22,
            positions=positions,
            showfliers=False,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5),
            flierprops=dict(markeredgewidth=1.5)
        )
        for i, (patch, color) in enumerate(zip(box['boxes'], box_colors)):
            patch.set_facecolor(color)
            if box_hatches[i]:
                patch.set_hatch(box_hatches[i])
        if not np.isnan(real_val):
            ax.axhline(real_val, color='#2ecc40', linestyle='--', linewidth=2)
        y_min = ax.get_ylim()[0]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.04
        for i, label in enumerate(xtick_labels):
            ax.text(positions[i], y_min, label, ha='center', va='top', fontsize=12, fontweight='bold', color='black', rotation=0)
        # Remove fermentation level text
        # y_min2 = ax.get_ylim()[0]-(ax.get_ylim()[1]-ax.get_ylim()[0])*0.12
        # ax.text(0.9, y_min2, 'VIS', ha='center', va='top', fontsize=13, color='#333333', fontweight='bold', rotation=0)
        # ax.text(2.1, y_min2, 'NIR', ha='center', va='top', fontsize=13, color='#333333', fontweight='bold', rotation=0)
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major', labelsize=13)
        if prop_idx == 0:
            ax.set_ylabel(f'{region}\nValue')
    # Remove suptitle (no title)
    plt.tight_layout(rect=[0, 0.13, 1, 0.95])
    # Add legend below all subplots
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False, fontsize=13)
    plt.subplots_adjust(bottom=0.18)
    plt.savefig('grafico_combinado_batches.png')
    plt.savefig('grafico_combinado_batches.svg')
    print("Combined graphic saved as grafico_combinado_batches.png and grafico_combinado_batches.svg")

def main():
    print("="*100)
    print("ANÁLISIS COMBINADO: DATOS INDUSTRIALES Y VALIDACIÓN BATCH 4")
    print("="*100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelos globales (una vez para ambos análisis)
    global_ml_models, global_dl_models, global_ml_model_files, global_dl_model_files = find_global_best_models(INDUSTRIAL_MODALITIES, device)
    
    print("\nModelos globales utilizados:")
    for mod in ["VIS", "NIR"]:
        print(f"{mod} ML: {global_ml_model_files[mod]}")
        print(f"{mod} DL: {global_dl_model_files[mod]}")
    
    # ===== CARGAR DATOS INDUSTRIALES =====
    industrial_data = {}
    print("\n--- CARGANDO DATOS INDUSTRIALES ---")
    for modality in INDUSTRIAL_MODALITIES:
        d = get_modality_data(modality, device)
        if d is not None:
            industrial_data[modality["name"]] = d
    
    # ===== CARGAR DATOS DE VALIDACIÓN =====
    validation_data = {}
    print("\n--- CARGANDO DATOS DE VALIDACIÓN BATCH 4 ---")
    for modality in VALIDATION_MODALITIES:
        d = get_modality_data(modality, device)
        if d is not None:
            validation_data[modality["name"]] = d
    
    # ===== GENERAR TABLA Y GRÁFICOS COMBINADOS =====
    if industrial_data and validation_data:
        print_combined_table(industrial_data, validation_data, global_ml_models, global_dl_models)
        create_combined_plots(industrial_data, validation_data, global_ml_models, global_dl_models)
    else:
        print("Error: No se pudieron cargar los datos necesarios")
    
    print("\nAnálisis completado. Se generaron la tabla y los gráficos combinados.")

if __name__ == "__main__":
    main()
