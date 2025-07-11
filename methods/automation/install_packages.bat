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
echo Checking for CUDA support on your system...
echo.

REM First, try to detect NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected! Checking CUDA version compatibility...
    
    REM Get CUDA version if available
    for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%i
    echo Detected CUDA Version: %CUDA_VERSION%
    
    REM Install PyTorch based on CUDA version
    if "%CUDA_VERSION%" geq "12.0" (
        echo Installing PyTorch with CUDA 12.1 support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if "%CUDA_VERSION%" geq "11.8" (
        echo Installing PyTorch with CUDA 11.8 support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo CUDA version too old or unsupported. Installing CPU-only PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
) else (
    echo NVIDIA GPU not detected or drivers not installed.
    echo Installing CPU-only PyTorch for maximum compatibility...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM Verify PyTorch installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully'); print(f'CUDA Available: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: PyTorch installation may have issues. Trying fallback installation...
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    python -c "import torch; print(f'Fallback PyTorch {torch.__version__} installed')"
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
echo Checking CUDA availability...
python -c "import warnings; warnings.filterwarnings('ignore', message='pkg_resources is deprecated'); import torch; print('CUDA Available:', torch.cuda.is_available())"

echo Package installation completed successfully.
exit /b 0
