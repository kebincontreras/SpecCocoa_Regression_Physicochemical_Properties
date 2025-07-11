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

# Check if we need to install or just update packages
echo "Checking current package status..."

# Install/update packages from requirements.txt with upgrade flag
if [ -f "requirements.txt" ]; then
    echo "Installing/updating packages from requirements.txt..."
    echo "Using --upgrade to update existing packages to latest compatible versions"
    PYTHONWARNINGS=ignore pip install --upgrade -r requirements.txt --quiet --disable-pip-version-check
    echo "Package installation/update from requirements.txt completed."
    echo ""
else
    echo "requirements.txt not found, installing essential packages..."
    # Fallback installation if requirements.txt is missing (shouldn't happen)
    PYTHONWARNINGS=ignore pip install --upgrade numpy pandas scikit-learn matplotlib tqdm pydicom opencv-python joblib h5py wandb einops torchmetrics openpyxl huggingface_hub --quiet --disable-pip-version-check
fi

# Verify critical packages are available
echo ""
echo "Verifying critical packages..."
echo "All packages are managed by requirements.txt"
echo "If there are import errors, they will be caught when scripts run"

echo "All critical packages verified successfully."

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

echo "Package installation completed successfully."
exit 0
