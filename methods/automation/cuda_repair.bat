@echo off
REM =============================================================================
REM Robust CUDA Detection and Repair Script for PyTorch
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================
echo   CUDA Diagnostics and Repair
echo ============================================
echo.

REM Step 1: Detect NVIDIA GPU
echo [1/5] Detecting GPU hardware...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
) else (
    echo ✗ NVIDIA GPU not detected or drivers not installed
    echo.
    echo POSSIBLE SOLUTIONS:
    echo 1. Install or update NVIDIA drivers from: https://www.nvidia.com/drivers
    echo 2. Check that GPU is enabled in BIOS/UEFI
    echo 3. Check GPU power connections
    echo.
    echo Continuing with CPU-only installation...
    goto :install_cpu
)

echo.

REM Step 2: Detect CUDA version
echo [2/5] Detecting CUDA version...
set CUDA_VERSION=
set CUDA_MAJOR=

REM Method 1: nvidia-smi
for /f "tokens=*" %%i in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do (
    for /f "tokens=3" %%j in ("%%i") do set CUDA_VERSION=%%j
)

if defined CUDA_VERSION (
    echo ✓ CUDA detected via nvidia-smi: %CUDA_VERSION%
    for /f "tokens=1 delims=." %%a in ("%CUDA_VERSION%") do set CUDA_MAJOR=%%a
) else (
    REM Method 2: nvcc if available
    nvcc --version >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=4" %%i in ('nvcc --version ^| findstr "release"') do (
            set CUDA_VERSION=%%i
            set CUDA_VERSION=!CUDA_VERSION:~0,-1!
            for /f "tokens=1 delims=." %%a in ("!CUDA_VERSION!") do set CUDA_MAJOR=%%a
        )
        echo ✓ CUDA detected via nvcc: !CUDA_VERSION!
    ) else (
        echo ⚠ CUDA not detected by standard methods
        echo → Assuming CUDA 11.8 compatible (most common)
        set CUDA_MAJOR=11
        set CUDA_VERSION=11.8
    )
)

echo CUDA Major Version: %CUDA_MAJOR%
echo.

REM Step 3: Remove previous PyTorch installation
echo [3/5] Cleaning previous PyTorch installation...
pip uninstall torch torchvision torchaudio -y >nul 2>&1

REM Step 4: Install appropriate PyTorch
echo [4/5] Installing optimized PyTorch for your GPU...

if %CUDA_MAJOR% geq 12 (
    echo → Installing PyTorch for CUDA 12.x
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    set TORCH_TYPE=CUDA 12.1
) else if %CUDA_MAJOR% geq 11 (
    echo → Installing PyTorch for CUDA 11.x
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    set TORCH_TYPE=CUDA 11.8
) else if %CUDA_MAJOR% geq 10 (
    echo → Old CUDA but compatible, using CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    set TORCH_TYPE=CUDA 11.8 (compatible)
) else (
    goto :install_cpu
)

goto :verify

:install_cpu
echo → Installing CPU-optimized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
set TORCH_TYPE=CPU-only

:verify
REM Step 5: Comprehensive verification
echo.
echo [5/5] Comprehensive installation verification...
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
        print('✓ Basic GPU test: SUCCESS')
    except Exception as e:
        print(f'✗ Basic GPU test: FAILED - {e}')
else:
    print('→ Using CPU mode (no GPU acceleration)')
    
# Basic CPU test
try:
    x = torch.rand(5, 3)
    y = x * 2
    print('✓ Basic CPU test: SUCCESS')
except Exception as e:
    print(f'✗ Basic CPU test: FAILED - {e}')

print('=' * 50)
"

if %errorlevel% neq 0 (
    echo.
    echo ⚠ ERROR: Verification failed. Installing fallback version...
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo ✓ PyTorch CPU installed as fallback
)

echo.
echo ============================================
echo   Repair completed
echo ============================================
echo Installation type: %TORCH_TYPE%
echo.
echo NEXT STEPS:
echo 1. If you see "CUDA Available: True" - Perfect! You'll have GPU acceleration
echo 2. If you see "CUDA Available: False" - Will work with CPU (slower)
echo 3. For persistent issues, run: methods\automation\pytorch_diagnostics.bat
echo.
pause
