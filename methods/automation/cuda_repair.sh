#!/bin/bash
# =============================================================================
# Robust CUDA Detection and Repair Script for PyTorch (Linux)
# =============================================================================

set -e  # Exit on any error

echo "============================================"
echo "   CUDA Diagnostics and Repair"
echo "============================================"
echo ""

# Step 1: Detect NVIDIA GPU
echo "[1/5] Detecting GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    else
        echo "NVIDIA GPU not detected or drivers not installed"
        echo ""
        echo "POSSIBLE SOLUTIONS:"
        echo "1. Install NVIDIA drivers: sudo apt install nvidia-driver-xxx"
        echo "2. Check that GPU is enabled in BIOS/UEFI"
        echo "3. Check GPU power connections"
        echo ""
        echo "Continuing with CPU-only installation..."
        INSTALL_TYPE="cpu"
    fi
else
    echo "nvidia-smi not found - no GPU drivers installed"
    echo "Continuing with CPU-only installation..."
    INSTALL_TYPE="cpu"
fi

echo ""

# Step 2: Detect CUDA version
echo "[2/5] Detecting CUDA version..."
CUDA_VERSION=""
CUDA_MAJOR=""

# Method 1: nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | awk '{print $1}')
fi

if [ ! -z "$CUDA_VERSION" ]; then
    echo "CUDA detected via nvidia-smi: $CUDA_VERSION"
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
else
    # Method 2: nvcc if available
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        if [ ! -z "$CUDA_VERSION" ]; then
            echo "CUDA detected via nvcc: $CUDA_VERSION"
            CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
        fi
    fi
fi

if [ -z "$CUDA_VERSION" ]; then
    echo "CUDA not detected - installing CPU version"
    INSTALL_TYPE="cpu"
    TORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
else
    echo "CUDA $CUDA_VERSION detected"
    
    # Step 3: Map CUDA version to PyTorch wheel
    echo "[3/5] Mapping CUDA version to PyTorch..."
    case $CUDA_MAJOR in
        "12")
            INSTALL_TYPE="cuda121"
            TORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            echo "Using CUDA 12.1 PyTorch wheel"
            ;;
        "11")
            INSTALL_TYPE="cuda118"
            TORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            echo "Using CUDA 11.8 PyTorch wheel"
            ;;
        *)
            echo "Unsupported CUDA version: $CUDA_VERSION"
            echo "  Falling back to CPU installation"
            INSTALL_TYPE="cpu"
            TORCH_INSTALL_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            ;;
    esac
fi

echo ""

# Step 4: Clean installation
echo "[4/5] Performing clean PyTorch installation..."

# Activate virtual environment if not already active
if [ ! -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment already active: $VIRTUAL_ENV"
else
    echo "Activating virtual environment..."
    source "$(dirname "$0")/../../Regressio_cocoa_venv/bin/activate"
fi

# Clean previous installations
echo "  Removing existing PyTorch installations..."
pip uninstall torch torchvision torchaudio -y &> /dev/null || true

# Clear pip cache
echo "  Clearing pip cache..."
pip cache purge &> /dev/null || true

# Upgrade pip
echo "  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo "  Installing PyTorch ($INSTALL_TYPE)..."
echo "  Command: $TORCH_INSTALL_CMD"
eval $TORCH_INSTALL_CMD

if [ $? -eq 0 ]; then
    echo "PyTorch installation completed"
else
    echo "PyTorch installation failed, trying fallback..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    INSTALL_TYPE="cpu-fallback"
fi

echo ""

# Step 5: Verification
echo "[5/5] Comprehensive installation verification..."
python -c "
import torch
import sys

print('=' * 50)
print('PYTORCH INSTALLATION REPORT')
print('=' * 50)
print(f'PyTorch Version: {torch.__version__}')
print(f'Python Version: {sys.version.split()[0]}')

# Check CUDA
cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')

if cuda_available:
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'GPU Devices: {torch.cuda.device_count()}')
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}')
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'    Compute Capability: {props.major}.{props.minor}')
    
    # Basic GPU test
    try:
        x = torch.rand(5, 3).cuda()
        y = x * 2
        print('Basic GPU test: SUCCESS')
    except Exception as e:
        print(f'Basic GPU test: FAILED - {e}')
else:
    print('Using CPU mode (no GPU acceleration)')
    
# Basic CPU test
try:
    x = torch.rand(5, 3)
    y = x * 2
    print('Basic CPU test: SUCCESS')
except Exception as e:
    print(f'Basic CPU test: FAILED - {e}')

print('=' * 50)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Verification failed. Installing fallback version..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo "PyTorch CPU installed as fallback"
    INSTALL_TYPE="cpu-fallback"
fi

echo ""
echo "============================================"
echo "   Repair completed"
echo "============================================"
echo "Installation type: $INSTALL_TYPE"
echo ""
echo "NEXT STEPS:"
echo "1. If you see 'CUDA Available: True' - Perfect! You'll have GPU acceleration"
echo "2. If you see 'CUDA Available: False' - Will work with CPU (slower)"
echo "3. For persistent issues, check NVIDIA drivers installation"
echo ""
