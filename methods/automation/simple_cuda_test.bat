@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Test Simple de CUDA
echo ============================================

echo [1] Verificando nvidia-smi...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo OK - nvidia-smi funciona
    
    echo.
    echo [2] Detectando version CUDA...
    
    REM Extract CUDA version using temporary file
    nvidia-smi | findstr "CUDA Version" > temp_cuda.txt 2>nul
    
    if exist temp_cuda.txt (
        for /f "tokens=9" %%j in (temp_cuda.txt) do (
            set CUDA_VER=%%j
            echo CUDA Version detectada: !CUDA_VER!
            
            REM Extract major version
            for /f "tokens=1 delims=." %%a in ("!CUDA_VER!") do (
                set CUDA_MAJOR=%%a
                echo CUDA Major: !CUDA_MAJOR!
                
                if !CUDA_MAJOR! geq 12 (
                    echo Recomendacion: PyTorch CUDA 12.1
                ) else if !CUDA_MAJOR! geq 11 (
                    echo Recomendacion: PyTorch CUDA 11.8  
                ) else (
                    echo Recomendacion: PyTorch CPU
                )
            )
        )
        del temp_cuda.txt
    ) else (
        echo ERROR - No se pudo detectar CUDA
    )
    
) else (
    echo ERROR - nvidia-smi no funciona
)

echo.
echo [3] Test PyTorch...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())" 2>nul

echo.
echo Test completado
pause
