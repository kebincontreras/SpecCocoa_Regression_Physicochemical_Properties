from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.tensorboard import SummaryWriter
from methods.metrics_2 import compute_metric_params, overall_accuracy, average_accuracy, kappa
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

svm_config = dict(
    C=1e5,
    kernel='rbf',
    gamma=1.
)

rfc_config = dict(
    n_estimators=1000,
    max_depth=20
)

mlp_config = dict(
    solver='adam',
    max_iter=1000,
    alpha=1e-3,
)

knn_config = dict(
    n_neighbors=3
)

CLASSIFIERS = dict(
    svm=[SVC, svm_config],
    rfc=[RandomForestClassifier, rfc_config],
    mlp=[MLPClassifier, mlp_config],
    knn=[KNeighborsClassifier, knn_config]
)


'''
def classify(model, train_dataset, test_dataset, save_name=None):
    X_train, Y_train = train_dataset["X"], train_dataset["Y"]
    X_test, Y_test = test_dataset["X"], test_dataset["Y"]

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')
    r2 = r2_score(Y_test, Y_pred, multioutput='raw_values')

    print(f"MSE: {mse}")
    print(f"R² Score: {r2}")

    return {"mse": mse, "r2": r2}'
'''



def build_regressor(name, params=None):
    if params is None:
        params = {}

    if name == "svr":
        return MultiOutputRegressor(SVR(**params))  
    elif name == "rfr":
        return RandomForestRegressor(**params)  
    elif name == "mlp":
        return MLPRegressor(**params)  
    elif name == "knnr":
        return KNeighborsRegressor(**params)  
    else:
        raise ValueError(f"Regressor {name} not supported")



def regress1(model, train_dataset, test_dataset, save_name=None):
    X_train, Y_train = train_dataset["X"], train_dataset["Y"]
    X_test, Y_test = test_dataset["X"], test_dataset["Y"]

    print(f"\nEntrenando modelo {type(model).__name__}...")
    model.fit(X_train, Y_train)  # Entrenar el modelo

    # 🔹 Predicción en TRAIN
    Y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    r2_train = r2_score(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    mae_train = mean_absolute_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()

    # 🔹 Predicción en TEST
    Y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    r2_test = r2_score(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    mae_test = mean_absolute_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()

    # Identificar cada variable de salida
    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

    # Diccionarios de métricas para TRAIN y TEST
    metrics = {
        "train": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_train)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_train)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_train)}
        },
        "test": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_test)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_test)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_test)}
        }
    }

    return train_dataset, test_dataset, metrics  # 🔹 Devuelve los datasets y métricas de train y test

import os
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def regress1(model, train_dataset, test_dataset, save_name=None):
    X_train, Y_train = train_dataset["X"], train_dataset["Y"]
    X_test, Y_test = test_dataset["X"], test_dataset["Y"]

    print(f"\nEntrenando modelo {type(model).__name__}...")
    model.fit(X_train, Y_train)  # Entrenar el modelo

    # 🔹 Predicción en TRAIN
    Y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    r2_train = r2_score(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    mae_train = mean_absolute_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()

    # 🔹 Predicción en TEST
    Y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    r2_test = r2_score(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    mae_test = mean_absolute_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()

    # Identificar cada variable de salida
    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

    # Diccionarios de métricas para TRAIN y TEST
    metrics = {
        "train": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_train)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_train)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_train)}
        },
        "test": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_test)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_test)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_test)}
        }
    }

    # 🔹 Guardar el modelo si se especifica un nombre
    if save_name:
        save_path = f"model/Machine_Learning/VIS/{save_name}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Modelo guardado en: {save_path}")

    return train_dataset, test_dataset, metrics  # 🔹 Devuelve los datasets y métricas de train y test


import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def regress(model, train_dataset, test_dataset, modality, save_name=None):
    X_train, Y_train = train_dataset["X"], train_dataset["Y"]
    X_test, Y_test = test_dataset["X"], test_dataset["Y"]

    print(f"\n🔹 Entrenando modelo {type(model).__name__} en {modality}...")
    model.fit(X_train, Y_train)  # Entrenar el modelo

    # 🔹 Predicción en TRAIN
    Y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    r2_train = r2_score(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    mae_train = mean_absolute_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()

    # 🔹 Predicción en TEST
    Y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    r2_test = r2_score(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    mae_test = mean_absolute_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()

    # Identificar cada variable de salida
    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

    # Diccionarios de métricas para TRAIN y TEST
    metrics = {
        "train": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_train)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_train)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_train)}
        },
        "test": {
            "mse": {label: mse for label, mse in zip(output_labels, mse_test)},
            "r2": {label: r2 for label, r2 in zip(output_labels, r2_test)},
            "mae": {label: mae for label, mae in zip(output_labels, mae_test)}
        }
    }

    # 🔹 Mostrar en consola el R² promedio
    r2_mean_train = np.mean(r2_train)
    r2_mean_test = np.mean(r2_test)
    print(f"📊 R² Train = {r2_mean_train:.4f}, R² Test = {r2_mean_test:.4f}")

    # 🔹 Guardar el modelo después del entrenamiento
    if save_name:
        SAVE_DIR = f"model/Machine_Learning/{modality}/"
        os.makedirs(SAVE_DIR, exist_ok=True)
        model_filename = os.path.join(SAVE_DIR, f"{save_name}_{r2_mean_test:.4f}.pkl")
        
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        print(f"✅ Modelo guardado en: {model_filename}")

    return train_dataset, test_dataset, metrics

