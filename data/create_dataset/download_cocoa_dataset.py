#!/usr/bin/env python3
import os
import requests
import shutil
import rarfile
from tqdm import tqdm

RAR_URL = "https://huggingface.co/datasets/kebincontreras/Spectral_signatures_of_cocoa_beans/resolve/main/Spectral_signatures_of_cocoa_beans.rar"

RAR_DIR = os.path.join("data", "raw_dataset")
RAR_FILENAME = os.path.join(RAR_DIR, "Spectral_signatures_of_cocoa_beans.rar")
EXTRACT_DIR = RAR_DIR
SUBFOLDER_TO_FLATTEN = "Spectral_signatures_of_cocoa_beans"

def download_file(url, dest):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Download error: {response.status_code}")
    total = int(response.headers.get('content-length', 0))

    with open(dest, 'wb') as file:
        bar = tqdm(desc=f"Downloading {os.path.basename(dest)}", total=total, unit='B', unit_scale=True)
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))
        bar.close()

def extract_rar(rar_path, extract_to):
    # Configure unrar.exe from methods/utils/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    unrar_path = os.path.join(project_root, "methods", "utils", "unrar.exe")
    
    if not os.path.exists(unrar_path):
        raise FileNotFoundError(f"unrar.exe not found at: {unrar_path}")
    
    rarfile.UNRAR_TOOL = unrar_path
    print(f"Using unrar: {unrar_path}")
    
    if not rarfile.is_rarfile(rar_path):
        raise ValueError(f"Not a valid RAR file: {rar_path}")

    with rarfile.RarFile(rar_path) as rf:
        print("Contents:")
        for f in rf.infolist():
            print("   └─", f.filename)
        rf.extractall(extract_to)

    print(f"Extracted to: {extract_to}")
    return True

def flatten_extracted_folder(parent_dir, subfolder_name):
    source_dir = os.path.join(parent_dir, subfolder_name)
    if not os.path.exists(source_dir):
        print(f"Folder not found: {source_dir}")
        return

    for item in os.listdir(source_dir):
        src = os.path.join(source_dir, item)
        dst = os.path.join(parent_dir, item)
        shutil.move(src, dst)
    os.rmdir(source_dir)
    print(f"Flattened directory: {parent_dir}")

def main():
    print("Downloading Cocoa Dataset")
    print("=" * 50)

    os.makedirs(RAR_DIR, exist_ok=True)

    # Check if dataset is already extracted
    extracted_folder = os.path.join(RAR_DIR, SUBFOLDER_TO_FLATTEN)
    if os.path.exists(extracted_folder):
        print(f"Dataset already extracted at: {extracted_folder}")
        print("Skipping download and extraction.")
        print("Dataset is ready to use.")
        return

    # Check if RAR file already exists
    if os.path.exists(RAR_FILENAME):
        print(f"RAR file already exists: {RAR_FILENAME}")
        print("Skipping download.")
    else:
        print(f"Downloading from: {RAR_URL}")
        download_file(RAR_URL, RAR_FILENAME)

    print(f"Extracting {RAR_FILENAME}...")
    extract_rar(RAR_FILENAME, EXTRACT_DIR)

    print(f"Organizing directory structure...")
    flatten_extracted_folder(RAR_DIR, SUBFOLDER_TO_FLATTEN)

    print("Dataset downloaded and ready to use.")

if __name__ == "__main__":
    main()