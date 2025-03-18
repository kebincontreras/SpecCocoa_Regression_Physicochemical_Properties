import os
import numpy as np
import scipy.io as sio
from einops import rearrange
import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py

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
        
        # Encontrar el valor máximo en la fila
        max_value = np.max(row)
        
        # Normalizar la fila
        if max_value > 0:  # Prevenir división por cero
            normalized_data[i, :] = row / max_value

    return normalized_data

def normalize_data_min_max_by_row(data):
    normalized_data = np.zeros_like(data, dtype=np.float64)

    for i in range(data.shape[0]):
        row = data[i, :]
        
        # Encontrar el valor mínimo y máximo en la fila
        min_value = np.min(row)
        max_value = np.max(row)
        
        # Rango de la fila
        range_value = max_value - min_value
        
        # Normalizar la fila
        if range_value > 0:  # Prevenir división por cero
            normalized_data[i, :] = (row - min_value) / range_value

    return normalized_data

'''

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std

    return X_train_standardized, X_test_standardized

def prepare_data1(dataset_name, split, seed, dl=False, dataset_params=None):
    if dataset_name == 'cocoa_public':
        data = sio.loadmat('data/cocoa_public/cocoa_public.mat')
        spec = data['data']
        label = data['label'].squeeze()

    elif dataset_name == 'cocoa_regression':  # Cargar datos desde archivos .h5 separados
        #train_file = 'data/train_cocoa_dataset_normalized.h5'
        #test_file = 'data/test_cocoa_dataset_normalized.h5'

        train_file = 'data/train_nir_cocoa_dataset_normalized.h5'
        test_file = 'data/test_nir_cocoa_dataset_normalized.h5'

        # Cargar dataset de entrenamiento
        with h5py.File(train_file, 'r') as f:
            X_train = f['spec'][:]  # Espectros de entrenamiento
            Y_train = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))

        # Cargar dataset de prueba desde el archivo `test_cocoa_dataset.h5`
        with h5py.File(test_file, 'r') as f:
            X_test = f['spec'][:]  # Espectros de prueba
            Y_test = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))

    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Normalizar los datos de entrada (pero NO las etiquetas)
    X_train, X_test = standardize_data(X_train, X_test)

    num_bands = X_train.shape[-1]
    num_outputs = Y_train.shape[-1]  # 4 variables de salida

    print("==============================================")
    print(f"x_train: shape={X_train.shape}, dtype={X_train.dtype}")
    print(f"y_train: shape={Y_train.shape}, dtype={Y_train.dtype}")
    print(f"x_test: shape={X_test.shape}, dtype={X_test.dtype}")
    print(f"y_test: shape={Y_test.shape}, dtype={Y_test.dtype}")
    print(f"Number of bands = {num_bands}")
    print(f"Number of outputs = {num_outputs}")
    print("==============================================")

    # Crear datasets
    train_dataset = dict(X=X_train.astype(np.float64), Y=Y_train.astype(np.float64))
    test_dataset = dict(X=X_test.astype(np.float64), Y=Y_test.astype(np.float64))

    return train_dataset, test_dataset, num_bands, num_outputs



def prepare_data(dataset_name, dl=False, dataset_params=None):
    if dataset_name == 'cocoa_regression':  # Asegurar que este dataset está correctamente escrito
        train_file = 'data/train_nir_cocoa_dataset_normalized.h5'
        test_file = 'data/test_nir_cocoa_dataset_normalized.h5'

        # Verificar que los archivos existen
        import os
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Archivo no encontrado: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Archivo no encontrado: {test_file}")

        # Cargar dataset de entrenamiento
        with h5py.File(train_file, 'r') as f:
            X_train = f['spec'][:]
            Y_train = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))

        # Cargar dataset de prueba
        with h5py.File(test_file, 'r') as f:
            X_test = f['spec'][:]
            Y_test = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))

    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    # Verificar que las variables no están vacías
    if X_train is None or X_test is None:
        raise ValueError("Error: X_train o X_test son None")

    # Normalizar los datos
    X_train, X_test = standardize_data(X_train, X_test)

    num_bands = X_train.shape[-1]
    num_outputs = Y_train.shape[-1]  # 4 variables de salida

    if dl:  # Para Deep Learning
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        train_dataset = TensorDataset(X_train, Y_train)

        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()
        test_dataset = TensorDataset(X_test, Y_test)

        train_loader = DataLoader(train_dataset, batch_size=dataset_params["batch_size"], shuffle=True,
                                  num_workers=dataset_params["num_workers"], pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=dataset_params["batch_size"], shuffle=False,
                                 num_workers=dataset_params["num_workers"], pin_memory=True)

        return train_loader, test_loader, num_bands, num_outputs

    else:
        return dict(X=X_train, Y=Y_train), dict(X=X_test, Y=Y_test), num_bands, num_outputs

'''

import os
import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_data(dataset_name, modality, dl=False, dataset_params=None):
    """
    Carga los datos en función de la modalidad seleccionada (NIR o VIS)
    """
    if modality == "NIR":
        train_file = "data/train_nir_cocoa_dataset_normalized.h5"
        test_file = "data/test_nir_cocoa_dataset_normalized.h5"
        save_dir = "model/Deep_Learning/NIR"
    elif modality == "VIS":
        train_file = "data/train_cocoa_dataset_normalized.h5"
        test_file = "data/test_cocoa_dataset_normalized.h5"
        save_dir = "model/Deep_Learning/VIS"
    else:
        raise ValueError("Modalidad no válida. Usa 'NIR' o 'VIS'")
    
    # Verificar que los archivos existen
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Archivo no encontrado: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Archivo no encontrado: {test_file}")
    
    # Cargar dataset de entrenamiento
    with h5py.File(train_file, 'r') as f:
        X_train = f['spec'][:]
        Y_train = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))
    
    # Cargar dataset de prueba
    with h5py.File(test_file, 'r') as f:
        X_test = f['spec'][:]
        Y_test = np.column_stack((f['cadmium'][:], f['fermentation_level'][:], f['moisture'][:], f['polyphenols'][:]))
    
    # Normalizar datos
    X_train, X_test = standardize_data(X_train, X_test)
    
    num_bands = X_train.shape[-1]
    num_outputs = Y_train.shape[-1]
    
    os.makedirs(save_dir, exist_ok=True)  # Crear directorio de guardado si no existe
    
    if dl:
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        train_dataset = TensorDataset(X_train, Y_train)
        
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=dataset_params["batch_size"], shuffle=True, num_workers=dataset_params["num_workers"], pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=dataset_params["batch_size"], shuffle=False, num_workers=dataset_params["num_workers"], pin_memory=True)
        
        return train_loader, test_loader, num_bands, num_outputs, save_dir
    else:
        return dict(X=X_train, Y=Y_train), dict(X=X_test, Y=Y_test), num_bands, num_outputs, save_dir

def standardize_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std
    return X_train_standardized, X_test_standardized


