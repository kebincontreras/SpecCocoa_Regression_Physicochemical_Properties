@echo off
REM =============================================================================
REM Package Installation - Original Working Version
REM =============================================================================

setlocal enabledelayedexpansion

REM Configure environment variables to suppress warnings
set PYTHONWARNINGS=ignore::UserWarning:lightning_utilities,ignore::FutureWarning:sklearn,ignore::DeprecationWarning:pkg_resources
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=Regressio_cocoa_venv

echo ============================================
echo   Installing Required Packages
echo ============================================

REM Activate virtual environment first
echo Activating virtual environment...
call "%ENV_NAME%\Scripts\activate.bat"

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Check for CUDA availability before installing PyTorch
echo Checking CUDA support on your system...
echo.

REM First, try to detect NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ NVIDIA GPU detected! Checking CUDA compatibility...
    
    REM Create temporary file to store CUDA version
    nvidia-smi | findstr /C:"CUDA Version" > temp_cuda.txt 2>nul
    
    set CUDA_VERSION=
    set CUDA_MAJOR=
    
    REM Read CUDA version from temporary file
    if exist temp_cuda.txt (
        for /f "tokens=9 delims= " %%j in (temp_cuda.txt) do (
            set CUDA_VERSION=%%j
            goto :cuda_found
        )
    )
    
    :cuda_found
    del temp_cuda.txt >nul 2>&1
    
    if defined CUDA_VERSION (
        echo CUDA Version detected: !CUDA_VERSION!
        REM Extract major version for comparison
        for /f "tokens=1 delims=." %%a in ("!CUDA_VERSION!") do set CUDA_MAJOR=%%a
        echo CUDA Major version: !CUDA_MAJOR!
    ) else (
        echo → Could not determine exact CUDA version. Assuming CUDA 11.8...
        set CUDA_VERSION=11.8
        set CUDA_MAJOR=11
    )
    
    REM Install PyTorch based on CUDA version
    if !CUDA_MAJOR! geq 12 (
        echo → Installing PyTorch with CUDA 12.1 support for GPU...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if !CUDA_MAJOR! geq 11 (
        echo → Installing PyTorch with CUDA 11.8 support for GPU...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else if !CUDA_MAJOR! geq 10 (
        echo → Old CUDA version but supported. Using CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo → CUDA too old. Installing CPU-optimized PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
) else (
    echo ✗ NVIDIA GPU not detected or drivers not installed.
    echo → Installing CPU-optimized PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM Verify PyTorch installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print('✓ PyTorch', torch.__version__, 'installed successfully')" 2>nul
if %errorlevel% neq 0 (
    echo ⚠ WARNING: PyTorch installation issue. Trying fallback installation...
    pip uninstall torch torchvision torchaudio -y >nul 2>&1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    python -c "import torch; print('✓ PyTorch', torch.__version__, 'fallback installed (CPU)')"
) else (
    python -c "import torch; cuda_available = torch.cuda.is_available(); print('✓ CUDA Available:', 'YES' if cuda_available else 'NO')" 2>nul
    python -c "import torch; cuda_available = torch.cuda.is_available(); print('✓ GPU Device:', torch.cuda.get_device_name(0) if cuda_available else 'CPU') if cuda_available else print('→ CPU mode (no GPU acceleration)')" 2>nul
)
echo.

REM Install other requirements
if exist "requirements.txt" (
    echo Installing other packages from requirements.txt...
    REM Use --no-deps to avoid reinstalling torch, then install deps
    pip install -r requirements.txt --no-deps
    pip install -r requirements.txt
    echo Package installation from requirements.txt completed.
) else (
    echo requirements.txt not found, installing essential packages...
    pip install numpy pandas scikit-learn matplotlib tqdm pydicom opencv-python
)

REM Suppress pkg_resources deprecation warnings
echo Configuring warning filters...
python -c "import warnings; warnings.filterwarnings('ignore', message='pkg_resources is deprecated'); print('Warning filters configured')"

REM Check CUDA availability
echo Final CUDA verification...
python -c "import warnings; warnings.filterwarnings('ignore', message='pkg_resources is deprecated'); import torch; cuda_available = torch.cuda.is_available(); print('CUDA Status:', '✓ ENABLED - GPU available' if cuda_available else '✗ DISABLED - CPU only')" 2>nul
python -c "import warnings; warnings.filterwarnings('ignore', message='pkg_resources is deprecated'); import torch; cuda_available = torch.cuda.is_available(); print('GPU Acceleration:', torch.cuda.get_device_name(0)) if cuda_available else print('Note: Training will use CPU (slower but functional)')" 2>nul

echo Package installation completed successfully.
exit /b 0
