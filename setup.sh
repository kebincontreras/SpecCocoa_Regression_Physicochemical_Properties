#!/bin/bash

echo "============================================"
echo "  SpecCocoa Regression Project Setup and Run"
echo "============================================"

# Variables
PROJECT_NAME="SpecCocoa_Regression"
ENV_NAME="Regressio_cocoa_venv"
REQUIREMENTS_FILE="requirements.txt"
PY_CMD="python3"

# Check Python
echo "Verificando Python..."
if ! $PY_CMD --version; then
    echo "Error: Python no encontrado"
    exit 1
fi

# Check pip
echo "Verificando pip..."
if ! $PY_CMD -m pip --version; then
    echo "Error: pip no encontrado"
    exit 1
fi

# Check venv module
echo "Verificando modulo venv..."
if ! $PY_CMD -m venv --help > /dev/null 2>&1; then
    echo "Error: modulo venv no disponible"
    exit 1
fi

# Create virtual environment
echo "Verificando entorno virtual: $ENV_NAME"
if [ -f "$ENV_NAME/bin/python" ]; then
    echo "Entorno virtual encontrado"
else
    echo "Creando nuevo entorno virtual..."
    if [ -d "$ENV_NAME" ]; then
        rm -rf "$ENV_NAME"
    fi
    $PY_CMD -m venv "$ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error al crear entorno virtual"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activando entorno virtual..."
source "$ENV_NAME/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error al activar entorno"
    exit 1
fi

# Check if main packages are installed
echo "Verificando paquetes principales..."
python -c "import torch, numpy, pandas, matplotlib, sklearn" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Instalando paquetes..."
    python -m pip install --upgrade pip

    # Detect CUDA
    echo "Detectando CUDA..."
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "Instalando PyTorch CPU..."
        pip install torch torchvision torchaudio
    else
        echo "Instalando PyTorch CUDA..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi

    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Instalando dependencias del proyecto..."
        pip install -r "$REQUIREMENTS_FILE"
    else
        echo "Archivo requirements.txt no encontrado"
    fi
else
    echo "Paquetes principales OK"
fi

echo ""
echo "============================================"
echo "  Dataset Preparation"
echo "============================================"

# Download dataset
echo "[1/7] Descargando dataset base..."
python data/create_dataset/download_cocoa_dataset.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to download dataset"
    exit 1
fi

# Check and extract dataset
echo "[2/7] Verificando y descomprimiendo dataset..."
EXTRACTION_SUCCESS=0

if [ -f data/raw_dataset/Spectral_signatures_of_cocoa_beans/*.xlsx ] 2>/dev/null || [ -f data/raw_dataset/*.xlsx ] 2>/dev/null; then
    echo "Dataset already extracted and verified"
    EXTRACTION_SUCCESS=1
else
    echo "Dataset needs extraction, checking for archive files..."
    
    # Check if there are any rar files to extract
    if ls data/*.rar 1> /dev/null 2>&1; then
        echo "Found RAR files, attempting extraction..."
        
        # Method 1: Try unrar
        if command -v unrar > /dev/null 2>&1; then
            echo "Found unrar, attempting extraction..."
            for rarfile in data/*.rar; do
                unrar x -o+ "$rarfile" data/ > /dev/null 2>&1
            done
            if [ -f data/raw_dataset/*.xlsx ] 2>/dev/null; then
                EXTRACTION_SUCCESS=1
            fi
        fi
        
        # Method 2: Try 7z if unrar failed
        if [ $EXTRACTION_SUCCESS -eq 0 ] && command -v 7z > /dev/null 2>&1; then
            echo "Trying 7z extraction..."
            for rarfile in data/*.rar; do
                7z x "$rarfile" -odata/ -y > /dev/null 2>&1
            done
            if [ -f data/raw_dataset/*.xlsx ] 2>/dev/null; then
                EXTRACTION_SUCCESS=1
            fi
        fi
        
        # Method 3: Try Python packages
        if [ $EXTRACTION_SUCCESS -eq 0 ]; then
            echo "Trying Python-based extraction..."
            pip install rarfile --quiet > /dev/null 2>&1
            python -c "import rarfile, os, glob; [rarfile.RarFile(f).extractall('data') for f in glob.glob('data/*.rar')]" > /dev/null 2>&1
            if [ -f data/raw_dataset/*.xlsx ] 2>/dev/null; then
                EXTRACTION_SUCCESS=1
            fi
        fi
    else
        echo "No RAR files found, checking if dataset is already properly extracted..."
        if [ -d "data/raw_dataset" ]; then
            echo "Dataset directory exists, assuming extraction is complete"
            EXTRACTION_SUCCESS=1
        fi
    fi
fi

if [ $EXTRACTION_SUCCESS -eq 0 ]; then
    echo "ERROR: Could not extract dataset. Please install unrar or 7z and try again."
    echo "You can install with: sudo apt-get install unrar or sudo apt-get install p7zip-full"
    exit 1
fi
echo "Dataset extraction completed successfully!"

# Generate training datasets
echo "[3/7] Generando datasets de entrenamiento NIR y VIS..."
python data/create_dataset/create_NIR2025_dataset.py
python data/create_dataset/create_VIS2025_dataset.py

# Generate validation datasets
echo "[4/7] Generando datasets de validacion NIR y VIS..."
python data/create_dataset/create_NIR2025_val_dataset.py
python data/create_dataset/create_VIS2025_val_dataset.py

# Generate test datasets
echo "[5/7] Generando datasets de test regionales NIR y VIS..."
python data/create_dataset/create_NIR2025_test_dataset.py
python data/create_dataset/create_VIS2025_test_dataset.py

# Normalize datasets
echo "[6/7] Normalizando datasets..."
python data/create_dataset/normalize_datasets.py

# Train and test models
echo "[7/7] Entrenando y probando modelos..."
python train.py

python test.py

echo ""
echo "============================================"
echo "   Ejecucion Completada"
echo "============================================"
