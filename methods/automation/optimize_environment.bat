@echo off
REM =============================================================================
REM System Optimization and Warning Suppression
REM =============================================================================

echo Configurando optimizaciones del sistema...

REM Suppress common Python warnings
set PYTHONWARNINGS=ignore::UserWarning
set PYTHONWARNINGS=ignore::ConvergenceWarning
set PYTHONWARNINGS=ignore::FutureWarning
set PYTHONWARNINGS=ignore::DeprecationWarning

REM Memory and CUDA optimizations
set CUDA_VISIBLE_DEVICES=""
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PYTORCH_NO_CUDA_MEMORY_CACHING=1
set OMP_NUM_THREADS=1

REM Disable pip version warnings
set PIP_DISABLE_PIP_VERSION_CHECK=1

echo Optimizations configured:
echo   - Python warnings suppressed
echo   - Memory configuration optimized
echo   - CUDA disabled to avoid warnings
echo   - Pip version check deshabilitado
echo.
