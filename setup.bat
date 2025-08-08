@echo off
echo ============================================
echo   SpecCocoa Regression Project Setup and Run
echo ============================================

REM Variables
set PROJECT_NAME=SpecCocoa_Regression
set ENV_NAME=Regressio_cocoa_venv
set REQUIREMENTS_FILE=requirements.txt
set PY_CMD=python

echo Verificando Python...
%PY_CMD% --version
if %errorlevel% neq 0 (
    echo Error: Python no encontrado
    pause
    exit /b 1
)

echo Verificando pip...
%PY_CMD% -m pip --version
if %errorlevel% neq 0 (
    echo Error: pip no encontrado
    pause
    exit /b 1
)

echo Verificando modulo venv...
%PY_CMD% -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: modulo venv no disponible
    pause
    exit /b 1
)

REM Crear entorno virtual
echo Verificando entorno virtual: %ENV_NAME%
if exist "%ENV_NAME%\Scripts\python.exe" goto check_packages

echo Creando nuevo entorno virtual...
if exist "%ENV_NAME%" rmdir /s /q "%ENV_NAME%"
%PY_CMD% -m venv "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Error al crear entorno virtual
    pause
    exit /b 1
)

:check_packages
echo Activando entorno virtual...
call "%ENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Error al activar entorno
    pause
    exit /b 1
)

echo Verificando paquetes principales...
python -c "import torch, numpy, pandas, matplotlib, sklearn" >nul 2>&1
if %errorlevel% neq 0 goto install_packages
echo Paquetes principales OK
goto run_project

:install_packages
echo Instalando paquetes...
python -m pip install --upgrade pip

echo Detectando CUDA...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalando PyTorch CPU...
    pip install torch torchvision torchaudio
) else (
    echo Instalando PyTorch CUDA...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

if exist "%REQUIREMENTS_FILE%" (
    echo Instalando dependencias del proyecto...
    pip install -r "%REQUIREMENTS_FILE%"
) else (
    echo Archivo requirements.txt no encontrado
)

:run_project
echo.
echo ============================================
echo   Dataset Preparation
echo ============================================

echo [1/7] Descargando dataset base...
python data\create_dataset\download_cocoa_dataset.py
if %errorlevel% neq 0 (
    echo Error: Failed to download dataset
    pause
    exit /b 1
)

echo [2/7] Verificando y descomprimiendo dataset...
REM Check if dataset needs extraction based on the presence of key files
set EXTRACTION_SUCCESS=0
if exist "data\raw_dataset\Spectral_signatures_of_cocoa_beans\*.xlsx" (
    echo Dataset already extracted and verified
    set EXTRACTION_SUCCESS=1
) else if exist "data\raw_dataset\*.xlsx" (
    echo Dataset already extracted and verified  
    set EXTRACTION_SUCCESS=1
) else (
    echo Dataset needs extraction, checking for archive files...
    
    REM Check if there are any rar files to extract
    if exist "data\*.rar" (
        echo Found RAR files, attempting extraction...
        
        REM Method 1: Try 7-Zip
        if exist "C:\Program Files\7-Zip\7z.exe" (
            echo Found 7-Zip, attempting extraction...
            "C:\Program Files\7-Zip\7z.exe" x "data\*.rar" -o"data\" -y >nul 2>&1
            if exist "data\raw_dataset\*.xlsx" set EXTRACTION_SUCCESS=1
        ) else if exist "C:\Program Files (x86)\7-Zip\7z.exe" (
            echo Found 7-Zip, attempting extraction...
            "C:\Program Files (x86)\7-Zip\7z.exe" x "data\*.rar" -o"data\" -y >nul 2>&1
            if exist "data\raw_dataset\*.xlsx" set EXTRACTION_SUCCESS=1
        )
        
        REM Method 2: Try WinRAR if 7-Zip failed
        if %EXTRACTION_SUCCESS%==0 (
            if exist "C:\Program Files\WinRAR\WinRAR.exe" (
                echo Trying WinRAR extraction...
                "C:\Program Files\WinRAR\WinRAR.exe" x "data\*.rar" "data\" >nul 2>&1
                if exist "data\raw_dataset\*.xlsx" set EXTRACTION_SUCCESS=1
            ) else if exist "C:\Program Files (x86)\WinRAR\WinRAR.exe" (
                echo Trying WinRAR extraction...
                "C:\Program Files (x86)\WinRAR\WinRAR.exe" x "data\*.rar" "data\" >nul 2>&1
                if exist "data\raw_dataset\*.xlsx" set EXTRACTION_SUCCESS=1
            )
        )
        
        REM Method 3: Try Python packages
        if %EXTRACTION_SUCCESS%==0 (
            echo Trying Python-based extraction...
            pip install rarfile --quiet >nul 2>&1
            python -c "import rarfile, os, glob; [rarfile.RarFile(f).extractall('data') for f in glob.glob('data/*.rar')]" >nul 2>&1
            if exist "data\raw_dataset\*.xlsx" set EXTRACTION_SUCCESS=1
        )
    ) else (
        echo No RAR files found, checking if dataset is already properly extracted...
        if exist "data\raw_dataset" (
            echo Dataset directory exists, assuming extraction is complete
            set EXTRACTION_SUCCESS=1
        )
    )
)

if %EXTRACTION_SUCCESS%==0 (
    echo ERROR: Could not extract dataset. Please install 7-Zip or WinRAR and try again.
    pause
    exit /b 1
)
echo Dataset extraction completed successfully!

echo [3/7] Generando datasets de entrenamiento NIR y VIS...
python data\create_dataset\create_NIR2025_dataset.py
python data\create_dataset\create_VIS2025_dataset.py

echo [4/7] Generando datasets de validacion NIR y VIS...
python data\create_dataset\create_NIR2025_val_dataset.py
python data\create_dataset\create_VIS2025_val_dataset.py

echo [5/7] Generando datasets de test regionales NIR y VIS...
python data\create_dataset\create_NIR2025_test_dataset.py
python data\create_dataset\create_VIS2025_test_dataset.py

echo [6/7] Normalizando datasets...
python data\create_dataset\normalize_datasets.py

echo [7/7] Entrenando y probando modelos...
python train.py

python test.py

echo.
echo ============================================
echo    Ejecucion Completada
echo ============================================
pause
