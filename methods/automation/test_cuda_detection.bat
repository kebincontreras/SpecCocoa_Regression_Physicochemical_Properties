@echo off
REM =============================================================================
REM Test rápido del script de instalación corregido
REM =============================================================================

echo ============================================
echo   Test de Detección CUDA Corregida
echo ============================================
echo.

echo [1] Ejecutando nvidia-smi...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ nvidia-smi funciona
    
    echo.
    echo [2] Detectando versión CUDA...
    
    REM Test the improved CUDA detection logic
    nvidia-smi | findstr /C:"CUDA Version" > temp_cuda_test.txt 2>nul
    
    set CUDA_VERSION=
    set CUDA_MAJOR=
    
    if exist temp_cuda_test.txt (
        for /f "tokens=9 delims= " %%j in (temp_cuda_test.txt) do (
            set CUDA_VERSION=%%j
            echo Versión extraída: %%j
            goto :found_version
        )
    )
    
    :found_version
    del temp_cuda_test.txt >nul 2>&1
    
    if defined CUDA_VERSION (
        echo ✓ CUDA_VERSION detectada: !CUDA_VERSION!
        for /f "tokens=1 delims=." %%a in ("!CUDA_VERSION!") do set CUDA_MAJOR=%%a
        echo ✓ CUDA_MAJOR: !CUDA_MAJOR!
        
        if !CUDA_MAJOR! geq 12 (
            echo → Recomendación: PyTorch CUDA 12.1
        ) else if !CUDA_MAJOR! geq 11 (
            echo → Recomendación: PyTorch CUDA 11.8
        ) else (
            echo → Recomendación: PyTorch CPU
        )
    ) else (
        echo ⚠ No se pudo detectar CUDA_VERSION
        echo → Usando fallback CUDA 11.8
    )
    
) else (
    echo ✗ nvidia-smi no funciona
)

echo.
echo [3] Test de PyTorch simple...
python -c "import torch; print('PyTorch versión:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo ⚠ PyTorch no está instalado o hay problemas
) else (
    echo ✓ PyTorch funciona correctamente
)

echo.
echo ============================================
echo   Test completado
echo ============================================
pause
