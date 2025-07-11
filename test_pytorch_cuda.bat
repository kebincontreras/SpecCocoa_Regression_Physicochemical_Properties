@echo off
echo ============================================
echo   Test PyTorch 2.0.0 + CUDA 11.8 Installation
echo ============================================

REM Create new clean environment
echo Creating clean virtual environment...
python -m venv test_env

REM Activate and install
call test_env\Scripts\activate.bat

echo Installing PyTorch 2.0.0 with CUDA 11.8...
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

echo Testing PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo Test completed.
pause
