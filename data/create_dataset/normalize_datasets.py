#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para normalizar todos los datasets de cacao
Crea versiones normalizadas de todos los archivos .h5 en la carpeta data/
"""

import h5py
import numpy as np
from pathlib import Path

# Factores de normalización para las etiquetas
NORM_FACTORS = {
    'fermentation_level': 100,
    'moisture': 10,
    'cadmium': 5.6,
    'polyphenols': 50
}

def normalize_labels(input_file, output_file):
    print(f"🔄 Procesando: {input_file.name}")
    try:
        with h5py.File(input_file, 'r') as h5_in:
            spectra = h5_in['spec'][:]
            fermentation = h5_in['fermentation_level'][:]
            moisture = h5_in['moisture'][:]
            cadmium = h5_in['cadmium'][:]
            polyphenols = h5_in['polyphenols'][:]

        fermentation_norm = fermentation / NORM_FACTORS['fermentation_level']
        moisture_norm = moisture / NORM_FACTORS['moisture']
        cadmium_norm = cadmium / NORM_FACTORS['cadmium']
        polyphenols_norm = polyphenols / NORM_FACTORS['polyphenols']

        with h5py.File(output_file, 'w') as h5_out:
            h5_out.create_dataset('spec', data=spectra, compression='gzip', compression_opts=9)
            h5_out.create_dataset('fermentation_level', data=fermentation_norm, compression='gzip', compression_opts=9)
            h5_out.create_dataset('moisture', data=moisture_norm, compression='gzip', compression_opts=9)
            h5_out.create_dataset('cadmium', data=cadmium_norm, compression='gzip', compression_opts=9)
            h5_out.create_dataset('polyphenols', data=polyphenols_norm, compression='gzip', compression_opts=9)

        print(f"✅ Guardado: {output_file.name}")
        print(f"   📊 Fermentación: {fermentation_norm.min():.3f} - {fermentation_norm.max():.3f}")
        print(f"   📊 Humedad: {moisture_norm.min():.3f} - {moisture_norm.max():.3f}")
        print(f"   📊 Cadmio: {cadmium_norm.min():.3f} - {cadmium_norm.max():.3f}")
        print(f"   📊 Polifenoles: {polyphenols_norm.min():.3f} - {polyphenols_norm.max():.3f}")

    except Exception as e:
        print(f"❌ Error en {input_file.name}: {str(e)}")

def normalize_all_datasets():
    data_dir = Path(__file__).parent / "data"
    patterns = ["train_*_cocoa_dataset.h5", "test_*_cocoa_dataset.h5", "TEST_*_cocoa_dataset.h5"]

    print("🚀 Iniciando normalización de datasets de cacao")
    print(f"📁 Carpeta: {data_dir.resolve()}")
    print("=" * 60)

    total = 0
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        print(f"🔍 Patrón '{pattern}': {len(files)} archivos")

        for input_file in files:
            if "_normalized" in input_file.name:
                continue

            output_file = input_file.with_name(input_file.stem + "_normalized.h5")

            if output_file.exists():
                print(f"⏭️ Ya existe: {output_file.name} → omitiendo.")
                continue

            normalize_labels(input_file, output_file)
            total += 1
            print()

    print("=" * 60)
    print(f"🎉 Total archivos procesados: {total}")

    normalized_files = sorted(data_dir.glob("*_normalized.h5"))
    if normalized_files:
        print("\n📋 Archivos generados:")
        for file in normalized_files:
            print(f"   ✓ {file.name}")
    else:
        print("\n⚠️ No se generaron archivos normalizados.")

if __name__ == "__main__":
    print("🧪 NORMALIZADOR DE DATASETS DE CACAO")
    print("=" * 60)
    normalize_all_datasets()
