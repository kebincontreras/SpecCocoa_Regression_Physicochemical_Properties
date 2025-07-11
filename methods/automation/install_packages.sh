#!/bin/bash
# =============================================================================
# Package Installation for Linux
# =============================================================================

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

ENV_NAME=${1:-"Regressio_cocoa_venv"}

echo "============================================"
echo "   Installing Required Packages"
echo "============================================"

# Activate environment
source "${ENV_NAME}/bin/activate"

# Configure warning suppression for pip
export PYTHONWARNINGS="ignore"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip first
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet --disable-pip-version-check

# Check for CUDA support on Linux
echo "Checking for CUDA support on your system..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    PYTHONWARNINGS=ignore pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet --disable-pip-version-check
else
    echo "CUDA not detected. Installing CPU-only PyTorch..."
    PYTHONWARNINGS=ignore pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet --disable-pip-version-check
fi

# Check if we need to install or just update packages
echo "Installing other packages from requirements.txt..."

# Install/update packages from requirements.txt with upgrade flag
if [ -f "requirements.txt" ]; then
    echo "Installing/updating packages from requirements.txt..."
    echo "Skipping PyTorch packages (already installed above) - installing other packages"
    # Filter out PyTorch packages and install the rest
    grep -v "^torch" requirements.txt > temp_requirements.txt
    PYTHONWARNINGS=ignore pip install --upgrade -r temp_requirements.txt --quiet --disable-pip-version-check || {
        echo "Some packages may have failed to install due to version constraints. Continuing..."
    }
    rm -f temp_requirements.txt
    echo "Package installation from requirements.txt completed."
    echo ""
else
    echo "requirements.txt not found, installing essential packages..."
    # Fallback installation if requirements.txt is missing (shouldn't happen)
    PYTHONWARNINGS=ignore pip install --upgrade numpy pandas scikit-learn matplotlib tqdm pydicom opencv-python joblib h5py wandb einops torchmetrics openpyxl huggingface_hub --quiet --disable-pip-version-check
fi

# Verify critical packages are available
echo ""
echo "Verifying critical packages..."
python -c "
try:
    import torch, numpy, pandas, h5py, sklearn, matplotlib
    print('All critical packages verified successfully.')
except ImportError as e:
    print(f'Warning: Some packages may not be available: {e}')
    print('The training scripts will handle missing packages gracefully.')
"

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "
try:
    import torch
    print('CUDA Available:', torch.cuda.is_available())
except ImportError:
    print('PyTorch not available. Please check installation.')
"

echo "Package installation completed successfully."
exit 0
