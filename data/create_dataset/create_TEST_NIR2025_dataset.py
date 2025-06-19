import os
import h5py
import joblib
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter, medfilt

from sklearn.decomposition import PCA
from datetime import datetime

from sklearn.linear_model import LinearRegression


# functions

def compute_sam(a, b):
    assert a.ndim == 2, "a must have two dimensions, if you only have one, please add an new dimension in the first place"
    assert b.ndim == 2, "b must have two dimensions, if you only have one, please add an new dimension in the first place"

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.matmul(a, b.T) / np.matmul(a_norm, b_norm.T), a_min=-1.0, a_max=1.0))


def compute_msc(spectra):
    B, L = spectra.shape  # Número de firmas y bandas espectrales
    mean_spectrum = np.mean(spectra, axis=0)  # Espectro de referencia promedio

    spectra_msc = np.zeros_like(spectra)

    for i in range(B):
        # Ajuste de regresión lineal: X_i = a * X_ref + b
        X_i = spectra[i, :].reshape(-1, 1)  # Convertimos a columna
        X_ref = mean_spectrum.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_ref, X_i)

        a = model.coef_[0, 0]  # Pendiente
        b = model.intercept_[0]  # Intersección

        # Aplicar la corrección MSC
        spectra_msc[i, :] = (spectra[i, :] - b) / a

    return spectra_msc


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# Dataset parameters
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

base_dir = "/home/enmartz/Jobs/cacao/Base_Datos_Cacao/Oficial_Cacao"
out_dir = os.path.join("built_datasets")
os.makedirs(out_dir, exist_ok=True)

# set variables

efficiency_range = [1100, 2000]  # nanometers (this is the spectral range of the data)
entrega1_white_scaling = 1.0  # this is the number of converyor belt spectral signature samples for sam metric
conveyor_belt_samples = 500  # for sam metric
angle_error = 0.25  # angle error between conveyor belt and cocoa signatures
max_num_samples = 500  # selected samples from lot with higher sam

cocoa_batch_size = 50  # guillotine methodology (number of cocoa bean samples)
cocoa_batch_samples = 2000  # number of batch samples (number of repetitions of cocoa_batch_size)

plot_num_samples = 500  # number of samples to plot
debug = False  # debug mode
debug_pca = True  # debug pca mode

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# Dataset initialization
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

# set variables for cocoa dataset

entrega_numbers = [5, 5, 5, 5]
ferm_levels = [44, 70, 87, 96]
colors = ['red', 'blue', 'green', 'orange']

# Lista de marcadores únicos
markers = ['o', 's', 'D', 'P']

pcas = {}
mean_pcas = {}

# set path to cocoa dataset

full_cocoa_paths = {
    'train': {
        0: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L1F44H096E240225C170325NIRTRAIFULL.mat",
            "B": "B1F44H096E240225C170325NIRTRAIFULL.mat",
            "N": "N1F44H096E240225C170325NIRTRAIFULL.mat",
            "E": "Entrega 5"},
        1: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L2F70H144E240225C170325NIRTRAIFULL.mat",
            "B": "B2F70H144E240225C170325NIRTRAIFULL.mat",
            "N": "N2F70H144E240225C170325NIRTRAIFULL.mat",
            "E": "Entrega 5"},
        2: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L3F87H192E240225C170325NIRTRAIFULL.mat",
            "B": "B3F87H192E240225C170325NIRTRAIFULL.mat",
            "N": "N3F87H192E240225C170325NIRTRAIFULL.mat",
            "E": "Entrega 5"},
        3: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L4F96H259E240225C170325NIRTRAIFULL.mat",
            "B": "B4F96H259E240225C170325NIRTRAIFULL.mat",
            "N": "N4F96H259E240225C170325NIRTRAIFULL.mat",
            "E": "Entrega 5"},
    },
    'test': {
        0: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L1F44H096E240225C170325NIRTESTFULL.mat",
            "B": "B1F44H096E240225C170325NIRTESTFULL.mat",
            "N": "N1F44H096E240225C170325NIRTESTFULL.mat",
            "E": "Entrega 5"},
        1: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L2F70H144E240225C170325NIRTESTFULL.mat",
            "B": "B2F70H144E240225C170325NIRTESTFULL.mat",
            "N": "N2F70H144E240225C170325NIRTESTFULL.mat",
            "E": "Entrega 5"},
        2: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L3F87H192E240225C170325NIRTESTFULL.mat",
            "B": "B3F87H192E240225C170325NIRTESTFULL.mat",
            "N": "N3F87H192E240225C170325NIRTESTFULL.mat",
            "E": "Entrega 5"},
        3: {"P": "18_01_2025/Optical_lab_spectral/NIR",
            "L": "L4F96H259E240225C170325NIRTESTFULL.mat",
            "B": "B4F96H259E240225C170325NIRTESTFULL.mat",
            "N": "N4F96H259E240225C170325NIRTESTFULL.mat",
            "E": "Entrega 5"},
    },
}

# load wavelengths

wavelenghts_path = '22_11_2024/Optical_lab_spectral/NIR'
wavelengths = next(
    v for k, v in loadmat(os.path.join(base_dir, wavelenghts_path, 'wavelengths_NIR.mat')).items() if
    not k.startswith('__')).squeeze()
# set threshold between 400 and 900 nm

efficiency_threshold = (efficiency_range[0] <= wavelengths) & (wavelengths <= efficiency_range[1])
wavelengths = wavelengths[efficiency_threshold]

# pca name

pca_name = (f'er{efficiency_range[0]}-{efficiency_range[1]}'
            f'_cbs{conveyor_belt_samples}'
            f'_ae{angle_error}'
            f'_mns{max_num_samples}'
            f'_cbs{cocoa_batch_size}'
            f'_cbs{cocoa_batch_samples}')

# load labels via pandas

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd_labels = pd.read_excel(os.path.join(base_dir, 'Labels.xlsx'))

# load and build dataset

for subset_name, lot_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")

    cocoa_bean_dataset = []
    label_dataset = []
    cocoa_bean_batch_mean_dataset = []
    label_batch_mean_dataset = []
    with h5py.File(f'TEST_{subset_name}_nir_cocoa_dataset.h5', 'w') as d:
        dataset = d.create_dataset('spec', shape=(0, len(wavelengths)), maxshape=(None, len(wavelengths)),
                                   chunks=(256, len(wavelengths)), dtype=np.float32)
        fermset = d.create_dataset('fermentation_level', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.uint8)
        moistset = d.create_dataset('moisture', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.float32)
        cadmiumset = d.create_dataset('cadmium', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.float32)
        polyset = d.create_dataset('polyphenols', (0, 1), maxshape=(None, 1), chunks=(256, 1), dtype=np.float32)


        # Append new data to dataset
        def append_to_dataset(dataset, new_data):
            current_shape = dataset.shape
            new_shape = (current_shape[0] + new_data.shape[0], current_shape[1])
            dataset.resize(new_shape)
            dataset[current_shape[0]:] = new_data


        for label, lot_filename in lot_filenames.items():
            print(f"Processing {lot_filename['E']} - {lot_filename['L']}")
            cocoa_path = os.path.join(base_dir, lot_filename['P'])

            white = next(
                v for k, v in loadmat(os.path.join(cocoa_path, lot_filename['B'])).items() if not k.startswith('__'))
            black = next(
                v for k, v in loadmat(os.path.join(cocoa_path, lot_filename['N'])).items() if not k.startswith('__'))
            lot = next(
                v for k, v in loadmat(os.path.join(cocoa_path, lot_filename['L'])).items() if not k.startswith('__'))[
                  1:]

            # apply efficiency threshold

            white = white[:, efficiency_threshold]
            black = black[:, efficiency_threshold]
            lot = lot[:, efficiency_threshold]

            if lot_filename['E'] in ['Entrega 1', 'Entrega 2']:
                lot = lot + black.mean(axis=0)[None, ...]

            if lot_filename['E'] in ['Entrega 1']:
                white_max = white.max(axis=1)
                white_max_index = white_max.argsort()[-10:]
                white = white[white_max_index]

            # process white and black

            white = white.mean(axis=0)[None, ...]
            black = black.mean(axis=0)[None, ...]

            # lot = (lot - black) / (white - black)
            # lot = lot / lot.max(axis=-1, keepdims=True)

            if debug:
                plt.figure(figsize=(8, 8))
                plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

                plt.subplot(3, 1, 1)
                plt.plot(wavelengths, white[::white.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('White')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 2)
                plt.plot(wavelengths, black[::black.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Black')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 3)
                plt.plot(wavelengths, lot[::lot.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Lot')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.tight_layout()
                plt.show()

            # get conveyor belt signatures

            conveyor_belt = lot[:conveyor_belt_samples, :]
            cc_distances = compute_sam(lot, conveyor_belt)
            lot_distances = cc_distances.min(axis=-1)
            sorted_indices = np.argsort(lot_distances)[::-1]  # from higher sam to lower
            selected_indices = np.sort(sorted_indices[:max_num_samples])
            selected_cocoa = lot[selected_indices, :]

            if debug:
                plt.figure(figsize=(8, 8))
                plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

                plt.subplot(3, 1, 1)
                plt.plot(wavelengths, conveyor_belt.T, alpha=0.5)
                plt.title('Conveyor Belt')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 2)
                plt.plot(np.sort(lot_distances))
                plt.axvline(x=lot_distances.shape[0] - max_num_samples, color='r', linestyle='--',
                            label=f'Threshold for {max_num_samples} samples')
                plt.title('Sorted Lot Distances')
                plt.xlabel('Lot Sample')
                plt.ylabel('SAM')
                plt.grid()
                plt.legend()

                plt.subplot(3, 1, 3)
                plt.plot(wavelengths, selected_cocoa[::selected_cocoa.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Selected Cocoa')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.tight_layout()
                plt.show()

            # get cocoa lot with reflectance

            selected_cocoa_reflectance = (selected_cocoa - black) / (white - black)
            selected_cocoa_reflectance = selected_cocoa_reflectance / selected_cocoa_reflectance.max(axis=-1,
                                                                                                     keepdims=True)
            # selected_cocoa_reflectance = selected_cocoa

            if debug:
                plt.figure(figsize=(8, 8))
                plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

                plt.subplot(3, 1, 1)
                plt.plot(wavelengths, white[::white.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('White')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 2)
                plt.plot(wavelengths, black[::black.shape[0] // plot_num_samples + 1].T, alpha=0.5)
                plt.title('Black')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Intensity')
                plt.grid()

                plt.subplot(3, 1, 3)
                plt.plot(wavelengths,
                         selected_cocoa_reflectance[::selected_cocoa_reflectance.shape[0] // plot_num_samples + 1].T,
                         alpha=0.5)
                plt.title('Selected Cocoa Reflectance')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Reflectance')
                plt.grid()

                plt.tight_layout()
                plt.show()

            # filtering

            # selected_cocoa_reflectance_median = savgol_filter(selected_cocoa_reflectance.copy(), 15, 2)
            selected_cocoa_reflectance_median = median_filter(selected_cocoa_reflectance.copy(), 10)
            # selected_cocoa_reflectance_median_msc = compute_msc(selected_cocoa_reflectance_median.copy())

            if debug:
                plt.figure(figsize=(8, 8))
                plt.suptitle(lot_filename['E'] + ' - ' + lot_filename['L'])

                plt.subplot(2, 1, 1)
                plt.plot(wavelengths,
                         selected_cocoa_reflectance[::selected_cocoa_reflectance.shape[0] // plot_num_samples + 1].T,
                         alpha=0.5)
                plt.title('Selected Cocoa Reflectance')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Reflectance')
                plt.grid()

                plt.subplot(2, 1, 2)
                plt.plot(wavelengths,
                         selected_cocoa_reflectance_median[
                         ::selected_cocoa_reflectance_median.shape[0] // plot_num_samples + 1].T,
                         alpha=0.5)
                plt.title('Selected Cocoa Reflectance Median')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Reflectance')
                plt.grid()

                # plt.subplot(3, 1, 3)
                # plt.plot(wavelengths,
                #          selected_cocoa_reflectance_median_msc[
                #          ::selected_cocoa_reflectance_median_msc.shape[0] // plot_num_samples + 1].T,
                #          alpha=0.5)
                # plt.title('Selected Cocoa Reflectance Median MSC')
                # plt.xlabel('Wavelength [nm]')
                # plt.ylabel('Reflectance')
                # plt.grid()

                plt.tight_layout()
                plt.show()

            # append to dataset

            cocoa_bean_dataset.append(selected_cocoa_reflectance_median)
            label_dataset.append(np.ones(selected_cocoa_reflectance_median.shape[0], dtype=int) * label)

            # shuffle and batch mean
            cocoa_bean_batch_mean_aux = []
            for i in range(cocoa_batch_samples):
                random_indices = np.random.choice(selected_cocoa_reflectance_median.shape[0], cocoa_batch_size,
                                                  replace=False)
                cocoa_bean_batch_mean_aux.append(selected_cocoa_reflectance_median[random_indices].mean(axis=0))

            cocoa_bean_batch_mean_aux = np.stack(cocoa_bean_batch_mean_aux, axis=0)
            cocoa_bean_batch_mean_dataset.append(cocoa_bean_batch_mean_aux)
            label_batch_mean_dataset.append(np.ones(cocoa_bean_batch_mean_aux.shape[0], dtype=int) * label)

            # save dataset

            append_to_dataset(dataset, cocoa_bean_batch_mean_aux)

            ones_vector = np.ones((cocoa_bean_batch_mean_aux.shape[0], 1), dtype=int)
            gt_label = pd_labels[pd_labels['Lot'].str.contains(lot_filename['L'][:12])]
            append_to_dataset(fermset, ones_vector * gt_label['Fermentation'].values[0])
            append_to_dataset(moistset, ones_vector * gt_label['Moisture'].values[0])
            append_to_dataset(cadmiumset, ones_vector * float(gt_label['Cadmium'].values[0]))
            append_to_dataset(polyset, ones_vector * gt_label['Polyphenols'].values[0])

    # compute mean and std of dataset and plot

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = cocoa_bean_dataset[i]
            mean = X_class.mean(axis=0)
            std = X_class.std(axis=0)
            plt.plot(wavelengths, mean, color=colors[i], label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
            plt.fill_between(wavelengths, mean - std, mean + std, alpha=0.2, color=colors[i], linewidth=0.0)

        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        plt.title('Mean and std of Cocoa dataset by class')

        plt.grid()
        plt.tight_layout()
        plt.show()

    # compute PCA with X_mean using sklearn

    full_cocoa_bean_dataset = np.concatenate(cocoa_bean_dataset, axis=0)
    full_label_dataset = np.concatenate(label_dataset, axis=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    full_cocoa_bean_dataset = scaler.fit_transform(full_cocoa_bean_dataset)

    # Load the PCA model
    pca = joblib.load(f'pca_nir_{pca_name}.pkl')
    X_pca = pca.transform(full_cocoa_bean_dataset)

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = X_pca[full_label_dataset.squeeze() == i]
            plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, marker=markers[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        # plt.xlabel('Component 1: {:.2f}%'.format(explained_variance[0] * 100))
        # plt.ylabel('Component 2: {:.2f}%'.format(explained_variance[1] * 100))

        plt.title(f'Cocoa dataset PCA')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    # plot cocoa bean batch mean dataset

    plt.figure(figsize=(6, 5))
    for i in range(len(cocoa_bean_batch_mean_dataset)):
        X_class = cocoa_bean_batch_mean_dataset[i]
        mean = X_class.mean(axis=0)
        std = X_class.std(axis=0)
        plt.plot(wavelengths, mean, color=colors[i], label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')
        plt.fill_between(wavelengths, mean - std, mean + std, alpha=0.2, color=colors[i], linewidth=0.0)

    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.title('Mean of Cocoa dataset Batch Mean by class')

    plt.grid()
    plt.tight_layout()
    plt.show()

    # compute PCA with X_mean using sklearn

    full_cocoa_bean_batch_mean_dataset = np.concatenate(cocoa_bean_batch_mean_dataset, axis=0)
    full_label_batch_mean_dataset = np.concatenate(label_batch_mean_dataset, axis=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    full_cocoa_bean_batch_mean_dataset = scaler.fit_transform(full_cocoa_bean_batch_mean_dataset)

    # Load the PCA model
    pca = joblib.load(f'pca_mean_nir_{pca_name}.pkl')
    X_pca = pca.transform(full_cocoa_bean_batch_mean_dataset)

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = X_pca[full_label_batch_mean_dataset.squeeze() == i]
            plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, marker=markers[i],
                        label=f'E{entrega_numbers[i]}-F{ferm_levels[i]}')

        # plt.xlabel('Component 1: {:.2f}%'.format(explained_variance[0] * 100))
        # plt.ylabel('Component 2: {:.2f}%'.format(explained_variance[1] * 100))

        plt.title(f'Cocoa dataset PCA')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
