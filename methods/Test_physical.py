import os
import json
import pickle
import numpy as np
import torch
from methods.dataloader import prepare_data
from methods.dl import build_regressor

# Factores de normalización
NORM_FACTORS = {
    'Fermentation Level': 100,
    'Moisture': 10,
    'Cadmium': 5.6,
    'Polyphenols': 50
}

# Propiedades en el orden correcto
OUTPUT_LABELS = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

def get_best_model(modality):
    """ Encuentra el mejor modelo de Machine Learning basado en R² """
    model_dir = f"model/Machine_Learning/{modality}/"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ No hay modelos guardados en {model_dir}")

    models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    
    if not models:
        raise FileNotFoundError(f"❌ No hay modelos en {model_dir}")

    # Ordenar modelos por R² en el nombre del archivo
    models.sort(key=lambda x: float(x.split("_")[-1][:-4]), reverse=True)
    best_model_path = os.path.join(model_dir, models[0])

    print(f"✅ Cargando mejor modelo de ML: {best_model_path}")
    
    with open(best_model_path, "rb") as f:
        best_model = pickle.load(f)
    
    return best_model

def get_best_dl_model(modality, device, num_bands, num_outputs):
    """ Encuentra el mejor modelo de Deep Learning basado en R² y carga sus hiperparámetros. """
    model_dir = f"model/Deep_Learning/{modality}/"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ No hay modelos guardados en {model_dir}")

    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    
    if not models:
        raise FileNotFoundError(f"❌ No hay modelos en {model_dir}")

    # Ordenar modelos por R² en el nombre del archivo
    models.sort(key=lambda x: float(x.split("_")[-1][:-4]), reverse=True)
    best_model_path = os.path.join(model_dir, models[0])
    json_path = best_model_path.replace(".pth", ".json")

    print(f"✅ Cargando mejor modelo de DL: {best_model_path}")

    # Extraer el nombre del modelo
    model_name = best_model_path.split("/")[-1].split("_")[0]

    # 🔹 Cargar los hiperparámetros desde JSON
    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            hyperparams = json.load(json_file)
        print(f"✅ Hiperparámetros cargados desde: {json_path}")
    else:
        raise FileNotFoundError(f"❌ No se encontró el archivo JSON {json_path}")

    # Construir el modelo con los hiperparámetros correctos
    model_dict = build_regressor(model_name, hyperparams, num_bands, num_outputs, device=device)
    model = model_dict["model"]

    # Cargar los pesos entrenados
    #model.load_state_dict(torch.load(best_model_path))
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    model.to(device)
    model.eval()

    return model

def predict_sample(modality):
    """ Carga una muestra de test y predice los valores con ML y DL """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Cargar datos de ML
    _, test_dataset_ml, _, _, _ = prepare_data("cocoa_regression", modality, dl=False)
    
    if test_dataset_ml is None:
        raise ValueError(f"❌ Error: No se pudo cargar el dataset de Machine Learning para {modality}. Verifica los archivos de datos.")

    X_test, Y_test = test_dataset_ml["X"], test_dataset_ml["Y"]

    # 🔹 Seleccionar una muestra aleatoria
    sample_idx = np.random.randint(0, X_test.shape[0])
    X_sample = X_test[sample_idx].reshape(1, -1)

    # 🔹 Obtener modelos
    best_ml_model = get_best_model(modality)

    # 🔹 Cargar datos para DL con dataset_params
    dl_params = {"batch_size": 1, "num_workers": 0}
    train_loader_dl, test_loader_dl, num_bands, num_outputs, _ = prepare_data("cocoa_regression", modality, dl=True, dataset_params=dl_params)

    if train_loader_dl is None or test_loader_dl is None:
        raise ValueError(f"❌ Error: No se pudo cargar el dataset de Deep Learning para {modality}. Verifica que los archivos `.h5` existan y sean correctos.")

    best_dl_model = get_best_dl_model(modality, device, num_bands, num_outputs)

    # 🔹 Predicción con ML
    Y_pred_ml = best_ml_model.predict(X_sample)[0]

    # 🔹 Predicción con DL
    X_sample_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
    with torch.no_grad():
        Y_pred_dl = best_dl_model(X_sample_tensor).cpu().numpy()[0]

    # 🔹 Desnormalizar predicciones
    Y_pred_ml_denorm = {label: pred * NORM_FACTORS[label] for label, pred in zip(OUTPUT_LABELS, Y_pred_ml)}
    Y_pred_dl_denorm = {label: pred * NORM_FACTORS[label] for label, pred in zip(OUTPUT_LABELS, Y_pred_dl)}

    # 🔹 Mostrar resultados
    print("\n📊 **Resultados de Predicción** 📊")
    print(f"🔹 Modalidad: {modality}")
    print(f"🔹 Firma seleccionada del Test (índice {sample_idx})")

    print("\n🔹 Predicción con Machine Learning:")
    for label, value in Y_pred_ml_denorm.items():
        print(f"  {label}: {value:.4f}")

    print("\n🔹 Predicción con Deep Learning:")
    for label, value in Y_pred_dl_denorm.items():
        print(f"  {label}: {value:.4f}")

if __name__ == "__main__":
    modality = input("Ingrese la modalidad para predecir (NIR o VIS): ").strip().upper()
    
    if modality not in ["NIR", "VIS"]:
        print("⚠️ Modalidad no válida. Debe ser 'NIR' o 'VIS'.")
    else:
        predict_sample(modality)
