@echo off
REM =============================================================================
REM PyTorch Compatibility Diagnostics for Windows
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================
echo   PyTorch Compatibility Diagnostics
echo ============================================

echo [1] System Information:
echo OS: %OS%
echo Processor: %PROCESSOR_ARCHITECTURE%
echo.

echo [2] Python Information:
python --version
python -c "import platform; print(f'Python Architecture: {platform.architecture()[0]}'); print(f'Platform: {platform.platform()}')"
echo.

echo [3] NVIDIA GPU Detection:
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ NVIDIA GPU detected
    echo NVIDIA Driver Information:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo.
    echo CUDA Version:
    nvidia-smi | findstr "CUDA Version"
) else (
    echo ✗ NVIDIA GPU not detected or drivers not installed
    echo → Will use CPU-only PyTorch
)
echo.

echo [4] Current PyTorch Installation:
python -c "try: import torch; print(f'✓ PyTorch Version: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}'); print(f'✓ CUDA Version: {torch.version.cuda}' if torch.cuda.is_available() else '→ Using CPU mode'); print(f'✓ GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''); x = torch.randn(3, 3); y = torch.mm(x, x); print('✓ Basic tensor operations work'); [x.cuda(), print('✓ CUDA operations work')] if torch.cuda.is_available() else None; except ImportError: print('✗ PyTorch not installed'); except Exception as e: print(f'✗ PyTorch error: {e}')"
echo.

echo [5] Memory Information:
wmic computersystem get TotalPhysicalMemory /format:value | findstr "="
echo.

echo [6] Recommended PyTorch Installation:
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%i
    if defined CUDA_VERSION (
        echo Based on your CUDA version (!CUDA_VERSION!):
        if "!CUDA_VERSION!" geq "12.0" (
            echo → pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ) else if "!CUDA_VERSION!" geq "11.8" (
            echo → pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ) else (
            echo → pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            echo   (CUDA version too old)
        )
    ) else (
        echo → pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
) else (
    echo → pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo   (No NVIDIA GPU detected)
)

echo.
echo [7] Common Issues and Solutions:
echo.
echo Issue: "ImportError: No module named torch"
echo Solution: Run the recommended installation command above
echo.
echo Issue: "CUDA out of memory"
echo Solution: Reduce batch size in configuration files
echo.
echo Issue: "RuntimeError: CUDA error: no kernel image is available"
echo Solution: Install CPU version instead
echo.
echo Issue: "ModuleNotFoundError: No module named 'torch'"
echo Solution: Make sure virtual environment is activated
echo.
echo Issue: PyTorch hangs or freezes
echo Solution: Use CPU version: pip install torch --index-url https://download.pytorch.org/whl/cpu
echo.

echo ============================================
echo   Diagnostics completed
echo ============================================
pause
