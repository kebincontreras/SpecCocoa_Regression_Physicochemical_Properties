@echo off
REM =============================================================================
REM PyTorch Auto-Repair Script for Windows Compatibility Issues
REM =============================================================================

setlocal enabledelayedexpansion

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=Regressio_cocoa_venv

echo ============================================
echo   PyTorch Auto-Repair Tool
echo ============================================

REM Activate environment
call "%ENV_NAME%\Scripts\activate.bat"

echo [1] Removing potentially problematic PyTorch installation...
pip uninstall torch torchvision torchaudio -y

echo.
echo [2] Clearing pip cache...
pip cache purge

echo.
echo [3] Detecting optimal PyTorch version for your system...

REM Check system architecture
python -c "import platform; arch = platform.architecture()[0]; print(f'Architecture: {arch}')"

REM Advanced NVIDIA detection
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected. Analyzing compatibility...
    
    REM Get GPU compute capability
    python -c "
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        compute_cap = result.stdout.strip()
        print(f'GPU Compute Capability: {compute_cap}')
        
        # Determine best CUDA version based on compute capability
        cap_major = float(compute_cap.split('.')[0]) if '.' in compute_cap else 0
        if cap_major >= 7.0:
            print('Recommendation: Use CUDA 11.8 or 12.1')
        elif cap_major >= 6.0:
            print('Recommendation: Use CUDA 11.8')
        else:
            print('Recommendation: Use CPU version (GPU too old)')
    else:
        print('Could not determine compute capability')
except Exception as e:
    print(f'Error checking compute capability: {e}')
"

    REM Try CUDA installation with fallback
    echo.
    echo [4] Installing PyTorch with CUDA support...
    echo Trying CUDA 12.1 first...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --timeout 60
    
    REM Test CUDA installation
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())" >nul 2>&1
    if %errorlevel% neq 0 (
        echo CUDA 12.1 failed. Trying CUDA 11.8...
        pip uninstall torch torchvision torchaudio -y
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --timeout 60
        
        python -c "import torch; print('CUDA available:', torch.cuda.is_available())" >nul 2>&1
        if %errorlevel% neq 0 (
            echo CUDA installations failed. Installing CPU version...
            pip uninstall torch torchvision torchaudio -y
            goto install_cpu
        )
    )
    
    echo PyTorch with CUDA installed successfully!
    goto verify_installation
    
) else (
    echo No NVIDIA GPU detected. Installing CPU-optimized version...
    
    :install_cpu
    echo.
    echo [4] Installing CPU-optimized PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --timeout 60
)

:verify_installation
echo.
echo [5] Verifying installation...
python -c "
import torch
import torchvision
import torchaudio
print(f'✓ PyTorch {torch.__version__} installed successfully')
print(f'✓ TorchVision {torchvision.__version__} installed')
print(f'✓ TorchAudio {torchaudio.__version__} installed')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'✓ CUDA Version: {torch.version.cuda}')
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')

# Test basic operations
try:
    x = torch.randn(100, 100)
    y = torch.mm(x, x)
    print('✓ CPU tensor operations work')
    
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = torch.mm(x_gpu, x_gpu)
        print('✓ GPU tensor operations work')
        
except Exception as e:
    print(f'✗ Error in tensor operations: {e}')
"

if %errorlevel% equ 0 (
    echo.
    echo ✓ PyTorch repair completed successfully!
    echo Your system is now ready for machine learning tasks.
) else (
    echo.
    echo ✗ PyTorch repair failed. Manual intervention required.
    echo Please check the error messages above and contact support.
    exit /b 1
)

echo.
echo [6] Performance recommendations:
echo - For training: Consider using smaller batch sizes if memory issues occur
echo - For inference: CPU version is sufficient for most tasks
echo - For large models: GPU with at least 8GB VRAM recommended

pause
