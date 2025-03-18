import h5py
import numpy as np
import os

# Factores de normalización
NORM_FACTORS = {
    'fermentation': 100,
    'moisture': 10,
    'cadmium': 5.6,  # 0.8 * 7
    'polyphenols': 50
}

def normalize_labels(train_file, test_file, output_train, output_test):
    # Cargar datos de entrenamiento
    with h5py.File(train_file, 'r') as h5_train:
        fermentation_train = h5_train['fermentation_level'][:]
        moisture_train = h5_train['moisture'][:]
        cadmium_train = h5_train['cadmium'][:]
        polyphenols_train = h5_train['polyphenols'][:]
        spectra_train = h5_train['spec'][:]

    # Cargar datos de prueba
    with h5py.File(test_file, 'r') as h5_test:
        fermentation_test = h5_test['fermentation_level'][:]
        moisture_test = h5_test['moisture'][:]
        cadmium_test = h5_test['cadmium'][:]
        polyphenols_test = h5_test['polyphenols'][:]
        spectra_test = h5_test['spec'][:]

    # Aplicar normalización con los valores fijos
    fermentation_train_norm = fermentation_train / NORM_FACTORS['fermentation']
    moisture_train_norm = moisture_train / NORM_FACTORS['moisture']
    cadmium_train_norm = cadmium_train / NORM_FACTORS['cadmium']
    polyphenols_train_norm = polyphenols_train / NORM_FACTORS['polyphenols']

    fermentation_test_norm = fermentation_test / NORM_FACTORS['fermentation']
    moisture_test_norm = moisture_test / NORM_FACTORS['moisture']
    cadmium_test_norm = cadmium_test / NORM_FACTORS['cadmium']
    polyphenols_test_norm = polyphenols_test / NORM_FACTORS['polyphenols']

    # Guardar los datos normalizados en nuevos archivos H5
    def save_h5(output_file, spec, fermentation, moisture, cadmium, polyphenols):
        with h5py.File(output_file, 'w') as h5_out:
            h5_out.create_dataset('spec', data=spec)
            h5_out.create_dataset('fermentation_level', data=fermentation)
            h5_out.create_dataset('moisture', data=moisture)
            h5_out.create_dataset('cadmium', data=cadmium)
            h5_out.create_dataset('polyphenols', data=polyphenols)
        print(f"✅ Datos normalizados guardados en: {output_file}")

    save_h5(output_train, spectra_train, fermentation_train_norm, moisture_train_norm, cadmium_train_norm, polyphenols_train_norm)
    save_h5(output_test, spectra_test, fermentation_test_norm, moisture_test_norm, cadmium_test_norm, polyphenols_test_norm)

# Rutas de los archivos
train_file = r"C:\Users\USUARIO\Documents\GitHub\En_ejecucion\speccocoa_2\data\train_cocoa_dataset.h5"
test_file = r"C:\Users\USUARIO\Documents\GitHub\En_ejecucion\speccocoa_2\data\test_cocoa_dataset.h5"

# Salida con los labels normalizados
output_train = r"C:\Users\USUARIO\Documents\GitHub\En_ejecucion\speccocoa_2\data\train_cocoa_dataset_normalized.h5"
output_test = r"C:\Users\USUARIO\Documents\GitHub\En_ejecucion\speccocoa_2\data\test_cocoa_dataset_normalized.h5"

# Ejecutar normalización
normalize_labels(train_file, test_file, output_train, output_test)
