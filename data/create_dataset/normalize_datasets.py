#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to normalize all cocoa datasets.
Creates normalized versions of all .h5 files in the data/ folder.
"""

import h5py
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw_dataset")

# Normalization factors for the labels
NORM_FACTORS = {
    'fermentation_level': 100,
    'moisture': 10,
    'cadmium': 5.6,
    'polyphenols': 50
}

def normalize_labels(input_file, output_file):
    print(f"🔄 Processing: {input_file.name}")
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

        print(f"✅ Saved: {output_file.name}")
        print(f"   📊 Fermentation: {fermentation_norm.min():.3f} - {fermentation_norm.max():.3f}")
        print(f"   📊 Moisture: {moisture_norm.min():.3f} - {moisture_norm.max():.3f}")
        print(f"   📊 Cadmium: {cadmium_norm.min():.3f} - {cadmium_norm.max():.3f}")
        print(f"   📊 Polyphenols: {polyphenols_norm.min():.3f} - {polyphenols_norm.max():.3f}")

    except Exception as e:
        print(f"❌ Error in {input_file.name}: {str(e)}")

def normalize_all_datasets():
    patterns = ["train_*_cocoa_dataset.h5", "test_*_cocoa_dataset.h5"]

    print("🚀 Starting cocoa dataset normalization")
    print(f"📁 Folder: {DATA_DIR.resolve()}")
    print("=" * 60)

    total = 0
    for pattern in patterns:
        files = list(DATA_DIR.glob(pattern))
        print(f"🔍 Pattern '{pattern}': {len(files)} files")

        for input_file in files:
            if "_normalized" in input_file.name:
                continue

            output_file = input_file.with_name(input_file.stem + "_normalized.h5")

            if output_file.exists():
                print(f"⏭️ Already exists: {output_file.name} → skipping.")
                continue

            normalize_labels(input_file, output_file)
            total += 1
            print()

    print("=" * 60)
    print(f"🎉 Total files processed: {total}")

    normalized_files = sorted(DATA_DIR.glob("*_normalized.h5"))
    if normalized_files:
        print("\n📋 Generated files:")
        for file in normalized_files:
            print(f"   ✓ {file.name}")
    else:
        print("\n⚠️ No normalized files were generated.")

if __name__ == "__main__":
    print("🧪 COCOA DATASET NORMALIZER")
    print("=" * 60)
    normalize_all_datasets()
