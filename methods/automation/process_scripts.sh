#!/bin/bash
# =============================================================================
# Processing Scripts Automation for Linux - Smart Version
# =============================================================================

ENV_NAME=${1:-"Regressio_cocoa_venv"}

echo "============================================"
echo "   Executing Processing Scripts"
echo "============================================"

# Activate environment
source "${ENV_NAME}/bin/activate"

# Configure warning suppression
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Package verification skipped - all dependencies managed by requirements.txt
echo "Package verification skipped - all dependencies managed by requirements.txt"
echo "If import errors occur, they will be reported by the individual scripts"
echo ""

# Check if NIR datasets exist
echo "Verifying NIR datasets..."
if [ -f "data/train_nir_cocoa_dataset.h5" ] && [ -f "data/test_nir_cocoa_dataset.h5" ]; then
    echo "NIR datasets already exist. Skipping create_NIR2025_dataset.py..."
else
    echo "Missing NIR datasets. Executing create_NIR2025_dataset.py..."
    echo "This process may take several minutes processing the dataset..."
    python -W ignore data/create_dataset/create_NIR2025_dataset.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed in create_NIR2025_dataset.py"
        exit 1
    fi
    echo "NIR datasets generated successfully."
fi
echo ""

# Check if VIS datasets exist
echo "Verifying VIS datasets..."
if [ -f "data/train_vis_cocoa_dataset.h5" ] && [ -f "data/test_vis_cocoa_dataset.h5" ]; then
    echo "VIS datasets already exist. Skipping create_VIS2025_dataset.py..."
else
    echo "Missing VIS datasets. Executing create_VIS2025_dataset.py..."
    echo "Processing VIS dataset..."
    python -W ignore data/create_dataset/create_VIS2025_dataset.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed in create_VIS2025_dataset.py"
        exit 1
    fi
    echo "VIS datasets generated successfully."
fi
echo ""

# Check if normalized datasets exist
echo "Verifying normalized datasets..."
if [ -f "data/train_nir_cocoa_dataset_normalized.h5" ] && [ -f "data/test_nir_cocoa_dataset_normalized.h5" ] && [ -f "data/train_vis_cocoa_dataset_normalized.h5" ] && [ -f "data/test_vis_cocoa_dataset_normalized.h5" ]; then
    echo "Normalized datasets already exist. Skipping normalize_datasets.py..."
else
    echo "Missing normalized datasets. Executing normalize_datasets.py..."
    echo "Normalizing all generated datasets..."
    python -W ignore data/create_dataset/normalize_datasets.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed in normalize_datasets.py"
        exit 1
    fi
    echo "Normalized datasets generated successfully."
fi

echo ""
echo "Data processing scripts completed successfully."
exit 0
