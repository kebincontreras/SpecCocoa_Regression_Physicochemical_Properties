@echo off
REM =============================================================================
REM Manual PyTorch Fix - Use only if automatic repair fails
REM =============================================================================

echo Manual PyTorch Installation Fix
echo ================================

REM Activate virtual environment
call Regressio_cocoa_venv\Scripts\activate.bat

REM Clean pip cache completely
echo Cleaning pip cache...
python -m pip cache purge

REM Upgrade pip to latest
echo Upgrading pip...
python -m pip install --upgrade pip

REM Update pip index
echo Updating pip index...
python -m pip install --upgrade setuptools wheel

REM Install PyTorch with specific index
echo Installing PyTorch...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Verify installation
echo Verifying installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

pause
