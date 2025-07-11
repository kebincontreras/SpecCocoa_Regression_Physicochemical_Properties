import os
import numpy as np
import scipy.io as sio
from einops import rearrange
import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py
import os
import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader


def process_indian_pines_dataset(hsi, label):
    hsi_vectorized = rearrange(hsi, 'm n l -> (m n) l').astype(float)
    label_vectorized = rearrange(label, 'm n -> (m n)')

    remove_indices = label_vectorized == 0
    hsi_vectorized = hsi_vectorized[~remove_indices]
    label_vectorized = label_vectorized[~remove_indices] - 1

    for band in range(hsi_vectorized.shape[-1]):
        band_min = hsi_vectorized[:, band].min()
        band_max = hsi_vectorized[:, band].max()
        hsi_vectorized[:, band] = (hsi_vectorized[:, band] - band_min) / (band_max - band_min)

    return hsi_vectorized, label_vectorized


def split_dataset(spec, label, split, seed):
    np.random.seed(seed)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for lab in range(np.max(label) + 1):
        label_indices = np.argwhere(label == lab).squeeze()
        np.random.shuffle(label_indices)

        split_train = int(split['train'] * len(label_indices))
        x_train.extend(spec[label_indices[:split_train]])
        x_test.extend(spec[label_indices[split_train:]])

        y_train.extend(label[label_indices[:split_train]])
        y_test.extend(label[label_indices[split_train:]])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def normalize_data_by_row(data):
    normalized_data = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[0]):
        row = data[i, :]

        # Find the maximum value in the row
        max_value = np.max(row)

        # Normalize the row
        if max_value > 0:  # Prevent division by zero
            normalized_data[i, :] = row / max_value

    return normalized_data


def normalize_data_min_max_by_row(data):
    normalized_data = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[0]):
        row = data[i, :]

        # Find the minimum and maximum value in the row
        min_value = np.min(row)
        max_value = np.max(row)

        # Rango de la fila
        range_value = max_value - min_value

        # Normalizar la fila
        if range_value > 0:  # Prevent division by zero
            normalized_data[i, :] = (row - min_value) / range_value

    return normalized_data


def prepare_data(dataset_name, modality, dl=False, dataset_params=None):
    """
    Load data based on the selected modality (NIR or VIS)
    """
    if modality == "NIR":
        train_file = "data/train_nir_cocoa_dataset_normalized.h5"
        test_file = "data/test_nir_cocoa_dataset_normalized.h5"
        save_dir = "model/Deep_Learning/NIR"

    elif modality == "VIS":
        train_file = "data/train_vis_cocoa_dataset_normalized.h5"
        test_file = "data/test_vis_cocoa_dataset_normalized.h5"
        save_dir = "model/Deep_Learning/VIS"

    else:
        raise ValueError("Invalid modality. Use 'NIR' or 'VIS'")

    # Verify that files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"File not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"File not found: {test_file}")

    # Load training dataset
    with h5py.File(train_file, 'r') as f:
        X_train = f['spec'][:]
        Y_train = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))

    # Load test dataset
    with h5py.File(test_file, 'r') as f:
        X_test = f['spec'][:]
        Y_test = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))

    # Normalize data
    # X_train, X_test = standardize_data(X_train, X_test)
    # X_train, X_test = normalize_data(X_train, X_test)
    # X_train, X_test = normalize_minmax(X_train, X_test)

    num_bands = X_train.shape[-1]
    num_outputs = Y_train.shape[-1]

    os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist

    if dl:
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        train_dataset = TensorDataset(X_train, Y_train)

        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()
        test_dataset = TensorDataset(X_test, Y_test)

        # Use pin_memory only if CUDA is available
        use_pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_dataset, batch_size=dataset_params["batch_size"], shuffle=True,
                                  num_workers=dataset_params["num_workers"], pin_memory=use_pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=dataset_params["batch_size"], shuffle=False,
                                 num_workers=dataset_params["num_workers"], pin_memory=use_pin_memory)

        return train_loader, test_loader, num_bands, num_outputs, save_dir
    else:
        return dict(X=X_train, Y=Y_train), dict(X=X_test, Y=Y_test), num_bands, num_outputs, save_dir


def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std
    return X_train_standardized, X_test_standardized


def normalize_data(X_train, X_test):
    X_train_normalized = X_train / np.max(X_train, axis=1, keepdims=True)
    X_test_normalized = X_test / np.max(X_test, axis=1, keepdims=True)
    return X_train_normalized, X_test_normalized


def normalize_minmax(X_train, X_test):
    X_min = np.min(X_train, axis=1, keepdims=True)
    X_max = np.max(X_train, axis=1, keepdims=True)

    X_train_normalized = (X_train - X_min) / (
                X_max - X_min + 1e-8)  # A small value is added to avoid divisions by 0

    X_min_test = np.min(X_test, axis=1, keepdims=True)
    X_max_test = np.max(X_test, axis=1, keepdims=True)

    X_test_normalized = (X_test - X_min_test) / (X_max_test - X_min_test + 1e-8)

    return X_train_normalized, X_test_normalized
