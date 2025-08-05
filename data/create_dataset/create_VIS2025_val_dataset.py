import os
import h5py
import joblib
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

from sklearn.decomposition import PCA

# Configurar matplotlib para modo no interactivo
plt.ion()
plt.rcParams['figure.max_open_warning'] = 0

# functions

def compute_sam(a, b):
    assert a.ndim == 2, ("a must have two dimensions, "
                         "if you only have one, please add an new dimension in the first place")
    assert b.ndim == 2, ("b must have two dimensions, "
                         "if you only have one, please add an new dimension in the first place")

    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.matmul(a, b.T) / np.matmul(a_norm, b_norm.T), a_min=-1.0, a_max=1.0))


# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# Dataset parameters
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#


base_dir = "data/raw_dataset/Spectral_signatures_of_cocoa_beans"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
wavelenghts_path = 'test/Optical_lab_spectral/VIS'

# set variables

efficiency_range = [500, 850]  # nanometers (this is the spectral range of the data)
conveyor_belt_samples = 200  # this is the number of converyor belt spectral signature samples for sam metric
angle_error = 0.25  # angle error between conveyor belt and cocoa signatures
max_num_samples = 1000  # selected samples from lot with higher sam

cocoa_batch_size = 50  # guillotine methodology (number of cocoa bean samples)
cocoa_batch_samples = 1000  # number of batch samples (number of repetitions of cocoa_batch_size)

plot_num_samples = 500  # number of samples to plot
debug = False  # debug mode
debug_pca = True  # debug pca mode

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# Dataset initialization
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

# set variables for cocoa dataset

entrega_numbers = ['Val1', 'Val2', 'Val3', 'Val4']
ferm_levels = [44, 70, 87, 96]  # fermentation levels
colors = ['red', 'blue', 'green', 'orange']

# Lista de marcadores únicos
markers = ['o', 's', 'D', 'P']

pcas = {}
mean_pcas = {}

# set path to cocoa dataset

full_cocoa_paths = {
    'train': {0: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L1F44H096E240225C140325VISTRAIFULL.mat",
                 "B": "B1F44H096E240225C140325VISTRAIFULL.mat",
                 "N": "N1F44H096E240225C140325VISTRAIFULL.mat",
                 "E": "Validation1"},
              1: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L2F70H144E240225C140325VISTRAIFULL.mat",
                 "B": "B2F70H144E240225C140325VISTRAIFULL.mat",
                 "N": "N2F70H144E240225C140325VISTRAIFULL.mat",
                 "E": "Validation2"},
              2: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L3F87H192E240225C140325VISTRAIFULL.mat",
                 "B": "B3F87H192E240225C140325VISTRAIFULL.mat",
                 "N": "N3F87H192E240225C140325VISTRAIFULL.mat",
                 "E": "Validation3"},
              3: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L4F96H259E240225C140325VISTRAIFULL.mat",
                 "B": "B4F96H259E240225C140325VISTRAIFULL.mat",
                 "N": "N4F96H259E240225C140325VISTRAIFULL.mat",
                 "E": "Validation4"},
             },
    'test': {0: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L1F44H096E240225C140325VISTESTFULL.mat",
                 "B": "B1F44H096E240225C140325VISTESTFULL.mat",
                 "N": "N1F44H096E240225C140325VISTESTFULL.mat",
                 "E": "Validation1"},
              1: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L2F70H144E240225C140325VISTESTFULL.mat",
                 "B": "B2F70H144E240225C140325VISTESTFULL.mat",
                 "N": "N2F70H144E240225C140325VISTESTFULL.mat",
                 "E": "Validation2"},
              2: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L3F87H192E240225C140325VISTESTFULL.mat",
                 "B": "B3F87H192E240225C140325VISTESTFULL.mat",
                 "N": "N3F87H192E240225C140325VISTESTFULL.mat",
                 "E": "Validation3"},
              3: {"P": "18_01_2025/Optical_lab_spectral/VIS",
                 "L": "L4F96H259E240225C140325VISTESTFULL.mat",
                 "B": "B4F96H259E240225C140325VISTESTFULL.mat",
                 "N": "N4F96H259E240225C140325VISTESTFULL.mat",
                 "E": "Validation4"},
             },
}

# load wavelengths

wavelengths = next(
    v for k, v in loadmat(os.path.join(base_dir, wavelenghts_path, 'wavelengths_VIS.mat')).items() if
    not k.startswith('__')).squeeze()

# set threshold between 400 and 900 nm

efficiency_threshold = (efficiency_range[0] <= wavelengths) & (wavelengths <= efficiency_range[1])
wavelengths = wavelengths[efficiency_threshold]

# load labels via pandas

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd_labels = pd.read_excel(os.path.join(base_dir, 'Labels.xlsx'))

# pca name

pca_name = (f'er{efficiency_range[0]}-{efficiency_range[1]}'
            f'_cbs{conveyor_belt_samples}'
            f'_ae{angle_error}'
            f'_mns{max_num_samples}'
            f'_cbs{cocoa_batch_size}'
            f'_cbs{cocoa_batch_samples}')

# load and build dataset

for subset_name, lot_filenames in full_cocoa_paths.items():
    print(f"Processing {subset_name} subset")

    cocoa_bean_dataset = []
    label_dataset = []
    cocoa_bean_batch_mean_dataset = []
    label_batch_mean_dataset = []
    with h5py.File(os.path.join(output_dir, f'{subset_name}_val_vis_cocoa_dataset.h5'), 'w') as d:
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

            white = white[:, efficiency_threshold.squeeze()]
            black = black[:, efficiency_threshold]
            lot = lot[:, efficiency_threshold]
            lot = np.delete(lot, 8719, axis=0) if lot_filename == 'L2F66H144R310324C070524VISTRAIFULL.mat' else lot

            # process white and black

            white = white.mean(axis=0)[None, ...]
            black = black.mean(axis=0)[None, ...]

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
                # plt.ylim([-100, 1000])
                plt.grid()

                plt.tight_layout()
                plt.show()
                plt.pause(2)
                plt.close()

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
                plt.pause(2)
                plt.close()

            # get cocoa lot with reflectance

            selected_cocoa_reflectance = (selected_cocoa - black) / (white - black)
            selected_cocoa_reflectance = selected_cocoa_reflectance / selected_cocoa_reflectance.max(axis=-1,
                                                                                                     keepdims=True)

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
                plt.pause(2)
                plt.close()

            # append to dataset

            cocoa_bean_dataset.append(selected_cocoa_reflectance)
            label_dataset.append(np.ones(selected_cocoa_reflectance.shape[0], dtype=int) * label)

            # shuffle and batch mean
            cocoa_bean_batch_mean_aux = []
            for i in range(cocoa_batch_samples):
                random_indices = np.random.choice(selected_cocoa_reflectance.shape[0], cocoa_batch_size, replace=False)
                cocoa_bean_batch_mean_aux.append(selected_cocoa_reflectance[random_indices].mean(axis=0))

            cocoa_bean_batch_mean_aux = np.stack(cocoa_bean_batch_mean_aux, axis=0)
            cocoa_bean_batch_mean_dataset.append(cocoa_bean_batch_mean_aux)
            label_batch_mean_dataset.append(np.ones(cocoa_bean_batch_mean_aux.shape[0], dtype=int) * label)

            # save dataset

            append_to_dataset(dataset, cocoa_bean_batch_mean_aux)

            ones_vector = np.ones((cocoa_bean_batch_mean_aux.shape[0], 1), dtype=int)
            search_key = lot_filename['L'].lstrip('L')  # Remove leading 'L' if already present
            gt_label = pd_labels[pd_labels['Lot'].str.contains('L' + search_key[:11])]
            append_to_dataset(fermset, ones_vector * gt_label['Fermentation'].values[0])
            append_to_dataset(moistset, ones_vector * gt_label['Moisture'].values[0])
            append_to_dataset(cadmiumset, ones_vector * float(str(gt_label['Cadmium'].values[0]).replace('<', '')))
            append_to_dataset(polyset, ones_vector * gt_label['Polyphenols'].values[0])

    # compute mean and std of dataset and plot

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = cocoa_bean_dataset[i]
            mean = X_class.mean(axis=0)
            std = X_class.std(axis=0)
            plt.plot(wavelengths, mean, color=colors[i], label=f'{entrega_numbers[i]}-F{ferm_levels[i]}')
            plt.fill_between(wavelengths, mean - std, mean + std, alpha=0.2, color=colors[i], linewidth=0.0)

        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Reflectance')
        plt.title('Mean and std of Cocoa dataset by class')

        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.pause(2)
        plt.close()

    # compute PCA with X_mean using sklearn

    full_cocoa_bean_dataset = np.concatenate(cocoa_bean_dataset, axis=0)
    full_label_dataset = np.concatenate(label_dataset, axis=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    full_cocoa_bean_dataset = scaler.fit_transform(full_cocoa_bean_dataset)
        
    pca = joblib.load(os.path.join(output_dir, f'pca_vis_{pca_name}.pkl'))
    X_pca = pca.transform(full_cocoa_bean_dataset)

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = X_pca[full_label_dataset.squeeze() == i]
            plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, marker=markers[i],
                        label=f'{entrega_numbers[i]}-F{ferm_levels[i]}')

        plt.title(f'Cocoa dataset PCA')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.pause(2)
        plt.close()

    # plot cocoa bean batch mean dataset

    plt.figure(figsize=(6, 5))
    for i in range(len(cocoa_bean_batch_mean_dataset)):
        X_class = cocoa_bean_batch_mean_dataset[i]
        mean = X_class.mean(axis=0)
        std = X_class.std(axis=0)
        plt.plot(wavelengths, mean, color=colors[i], label=f'{entrega_numbers[i]}-F{ferm_levels[i]}')
        plt.fill_between(wavelengths, mean - std, mean + std, alpha=0.2, color=colors[i], linewidth=0.0)

    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Reflectance')
    plt.title('Mean of Cocoa dataset Batch Mean by class')

    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.pause(2)
    plt.close()

    # compute PCA with X_mean using sklearn

    full_cocoa_bean_batch_mean_dataset = np.concatenate(cocoa_bean_batch_mean_dataset, axis=0)
    full_label_batch_mean_dataset = np.concatenate(label_batch_mean_dataset, axis=0)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    full_cocoa_bean_batch_mean_dataset = scaler.fit_transform(full_cocoa_bean_batch_mean_dataset)
    
    pca = joblib.load(os.path.join(output_dir, f'pca_mean_vis_{pca_name}.pkl'))
    X_pca = pca.transform(full_cocoa_bean_batch_mean_dataset)

    if debug_pca:
        plt.figure(figsize=(6, 5))
        for i in range(len(cocoa_bean_dataset)):
            X_class = X_pca[full_label_batch_mean_dataset.squeeze() == i]
            plt.scatter(X_class[:, 0], X_class[:, 1], color=colors[i], alpha=0.5, marker=markers[i],
                        label=f'{entrega_numbers[i]}-F{ferm_levels[i]}')

        plt.title(f'Cocoa dataset PCA')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.pause(2)
        plt.close()
