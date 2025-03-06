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

def regress(model_dict, train_loader, test_loader, save_name):
    model = model_dict["model"]
    criterion = model_dict["criterion"]
    optimizer = model_dict["optimizer"]
    
    device = next(model.parameters()).device
    model.train()

    num_epochs = model_dict["epochs"]
    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

    for epoch in range(1, num_epochs + 1):
        # 🔹 Inicializar almacenamiento de métricas
        metrics = {"epoch": epoch}
        mse_train, mse_test = [], []
        r2_train, r2_test = [], []
        mae_train, mae_test = [], []

        # 🔹 Entrenamiento
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

        # 🔹 Evaluación en Train y Test en cada época
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

            # 🔹 Calcular métricas
            mse_values = mean_squared_error(Y_true, Y_pred, multioutput='raw_values')
            r2_values = r2_score(Y_true, Y_pred, multioutput='raw_values')
            mae_values = mean_absolute_error(Y_true, Y_pred, multioutput='raw_values')

            # 🔹 Guardar en listas para impresión ordenada
            if dataset_name == "Train":
                mse_train, r2_train, mae_train = mse_values, r2_values, mae_values
            else:
                mse_test, r2_test, mae_test = mse_values, r2_values, mae_values

            # 🔹 Guardar métricas en el diccionario de W&B
            for label, mse, r2, mae in zip(output_labels, mse_values, r2_values, mae_values):
                metrics[f"{dataset_name}/MSE/{label}"] = mse
                metrics[f"{dataset_name}/R²/{label}"] = r2
                metrics[f"{dataset_name}/MAE/{label}"] = mae

        # 🔹 Calcular métricas globales
        mse_train_global, r2_train_global, mae_train_global = np.mean(mse_train), np.mean(r2_train), np.mean(mae_train)
        mse_test_global, r2_test_global, mae_test_global = np.mean(mse_test), np.mean(r2_test), np.mean(mae_test)

        # 🔹 Agregar métricas globales al diccionario
        metrics["Global/MSE_Train"] = mse_train_global
        metrics["Global/MSE_Test"] = mse_test_global
        metrics["Global/R²_Train"] = r2_train_global
        metrics["Global/R²_Test"] = r2_test_global
        metrics["Global/MAE_Train"] = mae_train_global
        metrics["Global/MAE_Test"] = mae_test_global

        # 🔹 Registrar métricas en W&B
        wandb.log(metrics)

        # 📌 Crear tabla con métricas individuales y globales
        table_data = []
        for label, mse_tr, mse_te, r2_tr, r2_te, mae_tr, mae_te in zip(output_labels, mse_train, mse_test, r2_train, r2_test, mae_train, mae_test):
            table_data.append([
                label,
                f"{mse_tr:.6f}", f"{mse_te:.6f}",
                f"{r2_tr:.6f}", f"{r2_te:.6f}",
                f"{mae_tr:.6f}", f"{mae_te:.6f}"
            ])

        # 📌 Agregar métricas globales a la tabla
        table_data.append([
            "🔹 Global",
            f"{mse_train_global:.6f}", f"{mse_test_global:.6f}",
            f"{r2_train_global:.6f}", f"{r2_test_global:.6f}",
            f"{mae_train_global:.6f}", f"{mae_test_global:.6f}"
        ])

        headers = ["Variable", "MSE Train", "MSE Test", "R² Train", "R² Test", "MAE Train", "MAE Test"]

        # 📌 Imprimir tabla con tabulate
        print(f"\n🔹 **Resultados de Evaluación - Epoch {epoch}** 🔹\n")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    return {
        "Train": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_train)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_train)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_train)}
        },
        "Test": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_test)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_test)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_test)}
        }
    }

def build_regressor(name, hyperparameters, num_bands, num_outputs, device):
    # Cargar configuración específica del modelo
    if name == "spectralnet":
        model = SpectralNet(num_bands, num_outputs, **config_spectralnet)
    elif name == "lstm":
        model = Lstm(num_bands, num_outputs, n_layers=config_lstm["n_layers"], dropout_rate=config_lstm["dropout_rate"])  # 🔹 Ahora usa la configuración correcta
    elif name == "cnn":
        model = CNN(num_bands, num_outputs, **config_cnn)
    elif name == "transformer":
        #model = TSTransformerEncoder(num_bands, num_outputs, **config_TSTransformer)
        model = SpectralFormer(input_dim=num_bands, **config_spectralformer, num_classes=num_outputs)
    elif name == "spectralformer":
        #model = SpectralFormer(num_bands, num_outputs, **config_spectralformer)
        model = SpectralFormer(input_dim=num_bands, **config_spectralformer, num_classes=num_outputs)
    else:
        raise ValueError(f"Regressor {name} not supported")

    # Enviar modelo a GPU o CPU
    model = model.to(device)

    # Definir función de pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "epochs": hyperparameters["epochs"],
    }

