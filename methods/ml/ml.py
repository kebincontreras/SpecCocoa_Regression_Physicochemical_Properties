from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import pickle
import numpy as np
import json


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


def regress(model, train_dataset, test_dataset, modality, save_name=None):
    X_train, Y_train = train_dataset["X"], train_dataset["Y"]
    X_test, Y_test = test_dataset["X"], test_dataset["Y"]

    print(f"\nTraining model {type(model).__name__} on {modality}...")
    model.fit(X_train, Y_train)  # Train the model

    # Prediction on TRAIN
    Y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    r2_train = r2_score(Y_train, Y_train_pred, multioutput='raw_values').tolist()
    mae_train = mean_absolute_error(Y_train, Y_train_pred, multioutput='raw_values').tolist()

    # Prediction on TEST
    Y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    r2_test = r2_score(Y_test, Y_test_pred, multioutput='raw_values').tolist()
    mae_test = mean_absolute_error(Y_test, Y_test_pred, multioutput='raw_values').tolist()

    # Identify each output variable
    output_labels = ["Cadmium", "Fermentation Level", "Moisture", "Polyphenols"]

    # Metrics dictionaries for TRAIN and TEST
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

    # Show average R² in console
    r2_mean_train = np.mean(r2_train)
    r2_mean_test = np.mean(r2_test)
    print(f"R² Train = {r2_mean_train:.4f}, R² Test = {r2_mean_test:.4f}")

    # Save the model and metrics after training
    if save_name:
        SAVE_DIR = f"model/Machine_Learning/{modality}/"
        os.makedirs(SAVE_DIR, exist_ok=True)
        model_filename = os.path.join(SAVE_DIR, f"{save_name}_{r2_mean_test:.4f}.pkl")
        metrics_filename = os.path.join(SAVE_DIR, f"{save_name}_{r2_mean_test:.4f}_metrics.json")

        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved at: {model_filename}")

        # Save the metrics to a JSON file
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: {metrics_filename}")

    return train_dataset, test_dataset, metrics



