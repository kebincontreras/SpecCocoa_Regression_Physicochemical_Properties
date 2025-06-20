#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para normalizar todos los datasets de cacao
Crea versiones normalizadas de todos los archivos .h5 existentes
"""

import h5py
import numpy as np
import os
import glob
import sys
from pathlib import Path

# Factores de normalización para las etiquetas
NORM_FACTORS = {
    'fermentation_level': 100,
    'moisture': 10,
    'cadmium': 5.6,  # 0.8 * 7
    'polyphenols': 50
}

def normalize_labels(input_file, output_file):
    """
    Normaliza las etiquetas de un archivo H5 y guarda el resultado
    
    Args:
        input_file (str): Ruta del archivo H5 original
        output_file (str): Ruta del archivo H5 normalizado a crear
    """
    print(f"🔄 Procesando: {os.path.basename(input_file)}")
    
    try:
        # Cargar datos originales
        with h5py.File(input_file, 'r') as h5_in:
            # Cargar espectros (no se normalizan)
            spectra = h5_in['spec'][:]
            
            # Cargar etiquetas
            fermentation = h5_in['fermentation_level'][:]
            moisture = h5_in['moisture'][:]
            cadmium = h5_in['cadmium'][:]
            polyphenols = h5_in['polyphenols'][:]
        
        # Aplicar normalización a las etiquetas
        fermentation_norm = fermentation / NORM_FACTORS['fermentation_level']
        moisture_norm = moisture / NORM_FACTORS['moisture']
        cadmium_norm = cadmium / NORM_FACTORS['cadmium']
        polyphenols_norm = polyphenols / NORM_FACTORS['polyphenols']
        
        # Guardar datos normalizados
        with h5py.File(output_file, 'w') as h5_out:
            # Guardar espectros sin cambios
            h5_out.create_dataset('spec', data=spectra, compression='gzip', compression_opts=9)
            
            # Guardar etiquetas normalizadas
            h5_out.create_dataset('fermentation_level', data=fermentation_norm, compression='gzip', compression_opts=9)
            h5_out.create_dataset('moisture', data=moisture_norm, compression='gzip', compression_opts=9)
            h5_out.create_dataset('cadmium', data=cadmium_norm, compression='gzip', compression_opts=9)
            h5_out.create_dataset('polyphenols', data=polyphenols_norm, compression='gzip', compression_opts=9)
        
        print(f"✅ Archivo normalizado guardado: {os.path.basename(output_file)}")
        
        # Mostrar estadísticas de normalización
        print(f"   📊 Estadísticas normalizadas:")
        print(f"      - Fermentación: {fermentation_norm.min():.3f} - {fermentation_norm.max():.3f}")
        print(f"      - Humedad: {moisture_norm.min():.3f} - {moisture_norm.max():.3f}")
        print(f"      - Cadmio: {cadmium_norm.min():.3f} - {cadmium_norm.max():.3f}")
        print(f"      - Polifenoles: {polyphenols_norm.min():.3f} - {polyphenols_norm.max():.3f}")
        
    except Exception as e:
        print(f"❌ Error procesando {input_file}: {str(e)}")

def normalize_all_datasets():
    """
    Normaliza todos los archivos de dataset encontrados en el directorio data/
    """
    # Directorio base donde están los datos - corregir la ruta
    data_dir = Path(__file__).parent.parent  # Esto apunta a data/
    
    # Patrones de archivos a normalizar
    patterns = [
        "train_*_cocoa_dataset.h5",
        "test_*_cocoa_dataset.h5",
        "TEST_*_cocoa_dataset.h5"
    ]
    
    print("🚀 Iniciando normalización de datasets de cacao")
    print("=" * 60)
    print(f"📁 Directorio de datos: {data_dir.absolute()}")
    print(f"📁 Buscando archivos en: {data_dir}")
    print()
    
    total_processed = 0
    
    for pattern in patterns:
        # Buscar archivos que coincidan con el patrón
        files = list(data_dir.glob(pattern))
        print(f"🔍 Patrón '{pattern}': {len(files)} archivos encontrados")
        
        for input_file in files:
            # Solo procesar si no es ya un archivo normalizado
            if "_normalized" not in input_file.name:
                # Crear nombre del archivo de salida en el mismo directorio data/
                output_name = input_file.stem + '_normalized.h5'
                output_file = data_dir / output_name
                
                print(f"📂 Guardando en: {output_file.absolute()}")
                
                # Normalizar archivo
                normalize_labels(str(input_file), str(output_file))
                total_processed += 1
                print()
    
    print("=" * 60)
    print(f"🎉 Normalización completada! {total_processed} archivos procesados")
    
    # Listar archivos normalizados creados
    normalized_files = list(data_dir.glob("*_normalized.h5"))
    if normalized_files:
        print("\n📋 Archivos normalizados disponibles:")
        for file in sorted(normalized_files):
            print(f"   ✓ {file.name} (en {file.parent})")
    else:
        print("\n⚠️ No se encontraron archivos normalizados")

def normalize_all_datasets_auto():
    """
    Normaliza todos los archivos de dataset sin interacción del usuario
    """
    # Directorio base donde están los datos
    data_dir = Path(__file__).parent.parent
    
    # Patrones de archivos a normalizar
    patterns = [
        "train_*_cocoa_dataset.h5",
        "test_*_cocoa_dataset.h5",
        "TEST_*_cocoa_dataset.h5"
    ]
    
    print("🚀 Iniciando normalización automática de datasets de cacao")
    print("=" * 60)
    print(f"📁 Directorio de datos: {data_dir}")
    print()
    
    total_processed = 0
    
    for pattern in patterns:
        # Buscar archivos que coincidan con el patrón
        files = list(data_dir.glob(pattern))
        
        for input_file in files:
            # Solo procesar si no es ya un archivo normalizado
            if "_normalized" not in input_file.name:
                # Crear nombre del archivo de salida
                name_parts = input_file.stem.split('_')
                output_name = '_'.join(name_parts) + '_normalized.h5'
                output_file = data_dir / output_name
                
                # Normalizar archivo
                normalize_labels(str(input_file), str(output_file))
                total_processed += 1
                print()
    
    print("=" * 60)
    print(f"🎉 Normalización completada! {total_processed} archivos procesados")
    
    # Listar archivos normalizados creados
    normalized_files = list(data_dir.glob("*_normalized.h5"))
    if normalized_files:
        print("\n📋 Archivos normalizados disponibles:")
        for file in sorted(normalized_files):
            print(f"   ✓ {file.name}")

def check_existing_files():
    """
    Verifica qué archivos existen y cuáles necesitan ser normalizados
    """
    data_dir = Path(__file__).parent.parent  # Directorio data/
    
    print("🔍 Verificando archivos existentes...")
    print("=" * 50)
    print(f"📁 Buscando en: {data_dir.absolute()}")
    
    # Archivos originales
    original_files = []
    patterns = ["train_*_cocoa_dataset.h5", "test_*_cocoa_dataset.h5", "TEST_*_cocoa_dataset.h5"]
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        for file in files:
            if "_normalized" not in file.name:
                original_files.append(file)
    
    print(f"📄 Archivos originales encontrados ({len(original_files)}):")
    for file in sorted(original_files):
        print(f"   • {file.name}")
        
        # Verificar si ya existe la versión normalizada
        normalized_name = file.stem + "_normalized.h5"
        normalized_path = data_dir / normalized_name
        
        if normalized_path.exists():
            print(f"     ✅ Ya normalizado: {normalized_name}")
        else:
            print(f"     ❌ Necesita normalización: {normalized_name}")
    
    print()

if __name__ == "__main__":
    print("🧪 NORMALIZADOR DE DATASETS DE CACAO")
    print("=" * 60)
    
    # Verificar archivos existentes
    check_existing_files()
    
    # Ejecutar normalización automáticamente sin preguntar
    normalize_all_datasets()
        
    # Verificar archivos existentes
    check_existing_files()

    normalize_all_datasets()
        