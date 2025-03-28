from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .models import ClassifierNet
from .models import SpectralNet, TSTransformerEncoder, Lstm, CNN, SpectralFormer
from .models.config import config_spectralnet, config_TSTransformer, config_lstm, config_cnn, config_spectralformer
from methods.models.config import config_lstm, config_cnn, config_TSTransformer, config_spectralnet, config_spectralformer
from methods.metrics_2 import compute_metric_params, overall_accuracy, average_accuracy, kappa
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb  
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabulate import tabulate  
import torch.nn as nn
import torch.optim as optim
import os
import json



BACKBONES = dict(
    spectralnet=[SpectralNet, config_spectralnet],
    cnn=[CNN, config_cnn],
    lstm=[Lstm, config_lstm],
    transformer=[TSTransformerEncoder, config_TSTransformer],
    spectralformer=[SpectralFormer, config_spectralformer]
)

def round_to_significant(x, sig=3):
    """ Redondea un número a tres cifras significativas """
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def build_regressor(model_name, hyperparameters, num_bands, num_outputs, device):
    if hyperparameters is None:
        raise ValueError("❌ Error: Los hiperparámetros no fueron proporcionados a build_regressor().")

    batch_size = hyperparameters.get("batch_size", 64)
    epochs = hyperparameters.get("epochs", 50)
    lr = hyperparameters.get("lr", 1e-4)
    weight_decay = hyperparameters.get("weight_decay", 1e-5)

    model_name = model_name.lower()
    if model_name in BACKBONES:
        model_class, base_config = BACKBONES[model_name]
        config_dict = base_config.copy()

        # ✅ Mapeo simple y uniforme
        config_dict["input_dim"] = num_bands
        config_dict["num_classes"] = num_outputs

        # Algunas arquitecturas usan feat_dim explícitamente
        if "feat_dim" in model_class.__init__.__code__.co_varnames:
            config_dict.setdefault("feat_dim", 1)

        model = model_class(**config_dict)
    else:
        raise ValueError(f"❌ Modelo '{model_name}' no está definido en BACKBONES.")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay
    }




def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, num_epochs, device):
    model.train()
    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

    print(f"🔹 Modelo en: {next(model.parameters()).device}")  # Debe imprimir 'cuda:0'
    
    for epoch in range(1, num_epochs + 1):
        metrics = {"epoch": epoch}
        mse_train, mse_test = [], []
        r2_train, r2_test = [], []
        mae_train, mae_test = [], []
        

        # Entrenamiento
        model.train()
        for X_batch, Y_batch in train_loader:

            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            print(f"X_batch en: {X_batch.device}, Y_batch en: {Y_batch.device}")  # Debería imprimir 'cuda:0'

            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

        # Evaluación en cada época
        for dataset_name, loader in [("Train", train_loader), ("Test", test_loader)]:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, Y_batch in loader:
                    X_batch = X_batch.to(device)
                    Y_pred = model(X_batch).cpu().numpy()
                    all_preds.append(Y_pred)
                    all_labels.append(Y_batch.cpu().numpy())

            Y_pred = np.vstack(all_preds)
            Y_true = np.vstack(all_labels)

            mse_values = mean_squared_error(Y_true, Y_pred, multioutput='raw_values')
            r2_values = r2_score(Y_true, Y_pred, multioutput='raw_values')
            mae_values = mean_absolute_error(Y_true, Y_pred, multioutput='raw_values')

            if dataset_name == "Train":
                mse_train, r2_train, mae_train = mse_values, r2_values, mae_values
            else:
                mse_test, r2_test, mae_test = mse_values, r2_values, mae_values

            for label, mse, r2, mae in zip(output_labels, mse_values, r2_values, mae_values):
                metrics[f"{dataset_name}/MSE/{label}"] = mse
                metrics[f"{dataset_name}/R²/{label}"] = r2
                metrics[f"{dataset_name}/MAE/{label}"] = mae

        wandb.log(metrics)
        print(f"Epoch {epoch}: R² Train = {np.mean(r2_train)}, R² Test = {np.mean(r2_test)}")
    
    return metrics




def regress(model_dict, train_loader, test_loader, model_name, modality):
    """
    Entrena y evalúa un modelo de Deep Learning.
    Guarda únicamente el mejor modelo según el R² promedio en test, incluyendo métricas en JSON.
    Si aparece un mejor modelo, elimina el anterior.
    """
    import os

    model = model_dict["model"]
    criterion = model_dict["criterion"]
    optimizer = model_dict["optimizer"]
    num_epochs = model_dict["epochs"]
    batch_size = model_dict["batch_size"]
    lr = model_dict["lr"]
    weight_decay = model_dict["weight_decay"]
    device = next(model.parameters()).device

    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

    best_r2 = -np.inf
    best_model_path = None
    best_json_path = None

    SAVE_DIR = f"model/Deep_Learning/{modality}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not wandb.run:
        wandb.init(project="2_nir_cocoa_regression_Deep_Learning", name=f"{model_name}_{modality}_experiment")

    for epoch in range(1, num_epochs + 1):
        metrics = {"epoch": epoch}

        # 🔹 Entrenamiento
        model.train()
        all_train_preds, all_train_labels = [], []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            all_train_preds.append(Y_pred.detach().cpu().numpy())
            all_train_labels.append(Y_batch.cpu().numpy())

        Y_train_pred = np.vstack(all_train_preds)
        Y_train_true = np.vstack(all_train_labels)
        r2_train = r2_score(Y_train_true, Y_train_pred, multioutput='raw_values')

        # 🔹 Evaluación
        model.eval()
        all_test_preds, all_test_labels = [], []
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device)
                Y_pred = model(X_batch).cpu().numpy()
                all_test_preds.append(Y_pred)
                all_test_labels.append(Y_batch.cpu().numpy())

        Y_test_pred = np.vstack(all_test_preds)
        Y_test_true = np.vstack(all_test_labels)
        r2_test = r2_score(Y_test_true, Y_test_pred, multioutput='raw_values')

        # 🔹 Registrar métricas
        for label, r2_t, r2_v in zip(output_labels, r2_train, r2_test):
            metrics[f"Train/R²/{label}"] = r2_t
            metrics[f"Test/R²/{label}"] = r2_v

        wandb.log(metrics)

        r2_mean_train = np.mean(r2_train)
        r2_mean_test = np.mean(r2_test)

        print(f"Epoch {epoch}/{num_epochs}: R² Train = {r2_mean_train:.4f}, R² Test = {r2_mean_test:.4f}")

        # 🔹 Guardar solo si mejora
        if r2_mean_test > best_r2:
            # Eliminar el modelo anterior si existe
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            if best_json_path and os.path.exists(best_json_path):
                os.remove(best_json_path)

            # Actualizar mejor R²
            best_r2 = r2_mean_test

            # Guardar nuevo modelo
            model_filename = os.path.join(SAVE_DIR, f"{model_name}_{best_r2:.4f}.pth")
            torch.save(model.state_dict(), model_filename)

            # Guardar hiperparámetros y métricas
            hyperparams = {
                "name": model_name,
                "batch_size": batch_size,
                "epochs": epoch,
                "lr": lr,
                "weight_decay": weight_decay,
                "test_r2": {label: float(r2_test[i]) for i, label in enumerate(output_labels)},
                "test_r2_mean": float(r2_mean_test)
            }
            json_filename = model_filename.replace(".pth", ".json")
            with open(json_filename, "w") as f:
                json.dump(hyperparams, f, indent=4)

            best_model_path = model_filename
            best_json_path = json_filename

            print(f"🆕 Modelo mejorado guardado: {model_filename}")

    return {
        "Best Test R²": best_r2
    }


