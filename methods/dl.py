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

def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, num_epochs, device):
    model.train()
    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]
    
    for epoch in range(1, num_epochs + 1):
        metrics = {"epoch": epoch}
        mse_train, mse_test = [], []
        r2_train, r2_test = [], []
        mae_train, mae_test = [], []

        # Entrenamiento
        model.train()
        for X_batch, Y_batch in train_loader:
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


def regress(model_dict, train_loader, test_loader, save_name):
    model = model_dict["model"]
    criterion = model_dict["criterion"]
    optimizer = model_dict["optimizer"]
    num_epochs = model_dict["epochs"]
    device = next(model.parameters()).device

    return train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, num_epochs, device)




