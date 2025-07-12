@echo off
REM =============================================================================
REM SpecCocoa Regression Project - Setup & Run (adaptado de run_project_GBM.bat)
REM =============================================================================

setlocal enabledelayedexpansion

REM Configuración del proyecto
set PROJECT_NAME=SpecCocoa_Regression
set ENV_NAME=Regressio_cocoa_venv
set REQUIREMENTS_FILE=requirements.txt
set AUTOMATION_DIR=methods\automation

echo ============================================
echo   SpecCocoa Regression Project Setup and Run
echo ============================================

REM ==============================
REM   Verificar Python
REM ==============================
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python no está instalado o no está en PATH.
    pause
    exit /b 1
)
echo Python detectado:
python --version

REM Obtener ruta de Python
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_PATH=%%i
echo Ejecutable de Python: %PYTHON_PATH%

REM Verificar pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip no está disponible. Intentando reparar...
    python -m ensurepip --upgrade --default-pip --user >nul 2>&1
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: No se pudo reparar pip.
        pause
        exit /b 1
    )
)

REM Verificar módulo venv
python -m venv --help >nul 2>&1
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
    python -m venv "%ENV_NAME%"
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

    REM Instalar PyTorch según disponibilidad de CUDA (igual que GBM)
    echo Verificando soporte CUDA...
    nvidia-smi >nul 2>&1
    if %errorlevel% equ 0 (
        echo CUDA detectado, instalando PyTorch con soporte CUDA...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo CUDA no detectado, instalando PyTorch CPU...
        pip install torch torchvision torchaudio
    )

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

REM Verificar dependencias principales
echo Verificando dependencias principales...
python -c "import torch, numpy, pandas, matplotlib, sklearn; print('Dependencias principales OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Faltan dependencias principales.
    pause
    exit /b 1
) else (
    echo Todas las dependencias principales están instaladas.
)

REM Crear carpetas necesarias
echo Creando carpetas necesarias...
if not exist "models" mkdir models
if not exist "figures" mkdir figures
if not exist "logs" mkdir logs
if not exist "data" mkdir data

REM ==============================
REM   Flujo principal del proyecto
REM ==============================
REM Step 4: Download Dataset
echo [4/7] Downloading Dataset...
call "%AUTOMATION_DIR%\download_dataset.bat"
if %errorlevel% neq 0 (
    echo Dataset download failed.
    pause
    exit /b 1
)
echo.

REM Step 5: Extract Dataset
echo [5/7] Extracting Dataset...
call "%AUTOMATION_DIR%\extract_dataset.bat" "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Dataset extraction failed.
    pause
    exit /b 1
)
echo.

REM Step 6: Process Scripts
echo [6/7] Processing Data Scripts...
call "%AUTOMATION_DIR%\process_scripts.bat" "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Data processing failed.
    pause
    exit /b 1
)
echo.

REM Step 7: Train and Test
echo [7/7] Training and Testing Models...
call "%AUTOMATION_DIR%\train_test.bat" "%ENV_NAME%"
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
