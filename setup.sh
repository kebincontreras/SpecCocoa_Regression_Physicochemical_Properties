#!/bin/bash
# =============================================================================
# SpecCocoa Regression Project - Simplified Linux Script
# =============================================================================

set -e  # Exit on any error

# Project configuration
PROJECT_NAME="SpecCocoa_Regression"
ENV_NAME="Regressio_cocoa_venv"
AUTOMATION_DIR="methods/automation"

echo "============================================"
echo "   SpecCocoa Regression Project Setup and Run"
echo "============================================"
echo "Starting automated setup process..."
echo ""

# Step 1: Check Python Environment
echo "[1/7] Checking Python Environment..."
bash "${AUTOMATION_DIR}/check_python.sh"
if [ $? -ne 0 ]; then
    echo "Python environment check failed."
    exit 1
fi
echo ""

# Step 2: Setup Virtual Environment
echo "[2/7] Setting up Virtual Environment..."
bash "${AUTOMATION_DIR}/setup_venv.sh" "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Virtual environment setup failed."
    exit 1
fi
echo ""

# Step 3: Install Packages
echo "[3/7] Installing Required Packages..."
bash "${AUTOMATION_DIR}/install_packages.sh" "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Package installation failed."
    exit 1
fi
echo ""

# Step 4: Download Dataset
echo "[4/7] Downloading Dataset..."
bash "${AUTOMATION_DIR}/download_dataset.sh"
if [ $? -ne 0 ]; then
    echo "Dataset download failed."
    exit 1
fi
echo ""

# Step 5: Extract Dataset
echo "[5/7] Extracting Dataset..."
bash "${AUTOMATION_DIR}/extract_dataset.sh" "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Dataset extraction failed."
    exit 1
fi
echo ""

# Step 6: Process Scripts
echo "[6/7] Processing Data Scripts..."
bash "${AUTOMATION_DIR}/process_scripts.sh" "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Data processing failed."
    exit 1
fi
echo ""

# Step 7: Train and Test
echo "[7/7] Training and Testing Models..."
bash "${AUTOMATION_DIR}/train_test.sh" "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Training and testing failed."
    exit 1
fi
echo ""

echo "============================================"
echo "   Execution Completed Successfully"
echo "============================================"
echo "All scripts executed correctly."
echo "The project is ready to use."
echo ""
echo "Generated files:"
echo "   - Trained models in: models/"
echo "   - Figures in: figures/"
echo "   - Logs in: logs/"
echo "   - Processed datasets in: data/"
echo ""
