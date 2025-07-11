@echo off
REM =============================================================================
REM Quick CUDA Diagnostics - SpecCocoa Project
REM =============================================================================

echo ============================================
echo   Quick GPU/CUDA Diagnostics
echo ============================================
echo.

echo [1] System Information:
echo OS: %OS%
echo Processor: %PROCESSOR_ARCHITECTURE%
echo.

echo [2] NVIDIA GPU Detection:
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ NVIDIA GPU detected
    echo.
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo.
    echo CUDA Driver Version:
    nvidia-smi | findstr "CUDA Version"
) else (
    echo ✗ NVIDIA GPU not detected
    echo → Check NVIDIA drivers installed
    echo → Download from: https://www.nvidia.com/drivers
)
echo.

echo [3] PyTorch Status (if installed):
python -c "
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f'✓ CUDA Available: YES')
        print(f'✓ CUDA Version: {torch.version.cuda}')
        print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('✗ CUDA Available: NO (using CPU)')
except ImportError:
    print('✗ PyTorch not installed')
except Exception as e:
    print(f'⚠ Error checking PyTorch: {e}')
" 2>nul
echo.

echo [4] Recommendations:
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✓ EVERYTHING PERFECT - GPU working correctly
    ) else (
        echo ⚠ GPU detected but PyTorch cannot use it
        echo → Run: methods\automation\cuda_repair.bat
        echo → Or reinstall PyTorch with CUDA support
    )
) else (
    echo → Install latest NVIDIA drivers
    echo → Check that GPU is enabled in BIOS
    echo → Check GPU power connections
)
echo.

echo ============================================
echo   Diagnostics completed
echo ============================================
pause
