@echo off
REM =============================================================================
REM PyTorch CUDA Installation Script
REM =============================================================================

setlocal enabledelayedexpansion

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=Regressio_cocoa_venv

echo ============================================
echo   Installing PyTorch with CUDA 11.8
echo ============================================

REM Activate virtual environment
call "%ENV_NAME%\Scripts\activate.bat"

echo Uninstalling existing PyTorch packages...
pip uninstall torch torchvision torchaudio -y

echo Installing PyTorch 2.2.0 with CUDA 11.8...
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

echo Forcing setuptools to correct version...
pip install setuptools==80.0.0 --force-reinstall --no-deps

echo Verifying PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo PyTorch with CUDA installation completed.
exit /b 0
