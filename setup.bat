@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM SpecCocoa Regression Project - Setup & Run
REM =============================================================================

set PROJECT_NAME=SpecCocoa_Regression
set ENV_NAME=Regressio_cocoa_venv
set REQUIREMENTS_FILE=requirements.txt

echo ============================================
echo   SpecCocoa Regression Project Setup and Run
echo ============================================

REM ==============================
REM   Find compatible Python (3.10, 3.9, 3.8 only)
REM ==============================
set PY_OK=0
set PY_CMD=

for %%V in (3.10 3.9 3.8) do (
    py -%%V --version > tmp_py_version.txt 2>&1
    findstr "Python %%V" tmp_py_version.txt >nul 2>&1
    if not errorlevel 1 (
        set PY_OK=1
        set PY_CMD=py -%%V
        goto :found_python
    )
)
del tmp_py_version.txt

:found_python
if %PY_OK%==0 (
    echo ==========================================================
    echo  ERROR: No compatible Python found (3.10, 3.9, or 3.8).
    echo  Please install Python 3.10 from: https://www.python.org/downloads/
    echo ==========================================================
    pause
    exit /b 1
)

echo Compatible Python found: !PY_CMD!
!PY_CMD! --version

REM Obtener ruta de Python
for /f "tokens=*" %%i in ('!PY_CMD! -c "import sys; print(sys.executable)"') do set PYTHON_PATH=%%i
echo Ejecutable de Python: !PYTHON_PATH!

REM Verificar pip
!PY_CMD! -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip no está disponible. Intentando reparar...
    !PY_CMD! -m ensurepip --upgrade --default-pip --user >nul 2>&1
    !PY_CMD! -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: No se pudo reparar pip.
        pause
        exit /b 1
    )
)

REM Verificar módulo venv
!PY_CMD! -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: El módulo venv no está disponible. Reinstala Python.
    pause
    exit /b 1
)

REM ==============================
REM   Manejo del entorno virtual
REM ==============================
set NEED_NEW_ENV=0
set NEED_INSTALL_PACKAGES=0

echo Verificando entorno virtual: %ENV_NAME%

if exist "%ENV_NAME%\Scripts\python.exe" (
    echo Carpeta del entorno encontrada
    call "%ENV_NAME%\Scripts\activate.bat" >nul 2>&1
    if %errorlevel% neq 0 (
        echo El entorno no se puede activar, se recreará.
        set NEED_NEW_ENV=1
    ) else (
        echo El entorno se puede activar
        "%ENV_NAME%\Scripts\python.exe" -c "import torch, numpy, pandas, matplotlib, sklearn; print('Paquetes principales OK')" >nul 2>&1
        if %errorlevel% neq 0 (
            echo Faltan paquetes requeridos
            set NEED_INSTALL_PACKAGES=1
        ) else (
            echo Todos los paquetes principales están instalados
            set NEED_INSTALL_PACKAGES=0
        )
        call deactivate >nul 2>&1
    )
) else (
    echo El entorno no existe, se creará uno nuevo.
    set NEED_NEW_ENV=1
    set NEED_INSTALL_PACKAGES=1
)

if %NEED_NEW_ENV%==1 (
    echo Creando nuevo entorno virtual...
    if exist "%ENV_NAME%" (
        echo Eliminando entorno anterior...
        rmdir /s /q "%ENV_NAME%" >nul 2>&1
    )
    !PY_CMD! -m venv "%ENV_NAME%"
    if %errorlevel% neq 0 (
        echo Error al crear el entorno virtual.
        pause
        exit /b 1
    )
    "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo pip no encontrado en el entorno, instalando...
        curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
        if exist "get-pip.py" (
            "%ENV_NAME%\Scripts\python.exe" get-pip.py >nul 2>&1
            del "get-pip.py" >nul 2>&1
        )
    )
    "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: No se pudo instalar pip en el entorno.
        pause
        exit /b 1
    )
    set NEED_INSTALL_PACKAGES=1
)

echo.
echo Activando entorno virtual...
call "%ENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Error al activar el entorno.
    pause
    exit /b 1
)
echo Entorno activado correctamente

if %NEED_INSTALL_PACKAGES%==1 (
    echo Instalando paquetes requeridos...
    python -m pip install --upgrade pip --quiet

    REM Detectar versión de CUDA y elegir el índice de PyTorch adecuado
    nvidia-smi | findstr "CUDA Version" > temp_cuda.txt 2>nul
    set CUDA_MAJOR=
    if exist temp_cuda.txt (
        for /f "tokens=9" %%j in (temp_cuda.txt) do (
            set CUDA_VER=%%j
            for /f "tokens=1 delims=." %%a in ("%%j") do set CUDA_MAJOR=%%a
        )
        del temp_cuda.txt
    )
    if "%CUDA_MAJOR%"=="" set CUDA_MAJOR=11

    echo CUDA_MAJOR detectado: %CUDA_MAJOR%

    setlocal EnableDelayedExpansion
    if "!CUDA_MAJOR!" GEQ "12" (
        echo Instalando PyTorch CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else (
        echo Instalando PyTorch CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    )
    endlocal

    REM Instalar requerimientos adicionales del proyecto
    if exist "%REQUIREMENTS_FILE%" (
        echo Instalando paquetes adicionales de %REQUIREMENTS_FILE%...
        pip install -r "%REQUIREMENTS_FILE%" --no-deps
        pip install -r "%REQUIREMENTS_FILE%"
    ) else (
        echo Archivo %REQUIREMENTS_FILE% no encontrado, instala manualmente tus dependencias.
    )
) else (
    echo Paquetes ya instalados, omitiendo instalación.
)

REM ==============================
REM   Flujo principal del proyecto
REM ==============================

echo [4/7] Downloading Dataset...
call methods\automation\download_dataset.bat
if %errorlevel% neq 0 (
    echo Dataset download failed.
    pause
    exit /b 1
)
echo.

echo [5/7] Extracting Dataset...
call methods\automation\extract_dataset.bat "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Dataset extraction failed.
    pause
    exit /b 1
)
echo.

echo [6/7] Processing Data Scripts...
call methods\automation\process_scripts.bat "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Data processing failed.
    pause
    exit /b 1
)
echo.

echo [7/7] Training and Testing Models...
call methods\automation\train_test.bat "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Training and testing failed.
    pause
    exit /b 1
)
echo.

echo ============================================
echo    Execution Completed Successfully
echo ============================================
echo All scripts executed correctly.
echo The project is ready to use.
echo.
echo Generated files:
echo   - Trained models in: models/
echo   - Figures in: figures/
echo   - Logs in: logs/
echo   - Processed datasets in: data/
echo.
pause
