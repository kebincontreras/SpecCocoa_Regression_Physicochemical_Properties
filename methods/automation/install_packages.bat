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
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo CUDA detected! Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo CUDA not detected. Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio
)

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
