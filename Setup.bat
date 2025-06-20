@echo off
chcp 65001 >nul

echo ============================================
echo CONFIGURACION DEL ENTORNO PYTHON
echo ============================================
echo.

REM Crear entorno virtual si no existe
if not exist "Regression_cocoa\Scripts\activate.bat" (
    echo Creando entorno virtual...
    python -m venv Regression_cocoa
)



REM Activar entorno virtual
echo Activando entorno virtual...
call Regression_cocoa\Scripts\activate.bat

REM Verificar que entorno está activo
echo Entorno virtual activo: %VIRTUAL_ENV%
python -c "import sys; print(' Ejecutando con Python en:', sys.executable)"
echo.

REM Instalar dependencias
echo Instalando dependencias desde requirements.txt...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Error instalando desde requirements.txt, intentando instalacion basica...
        python -m pip install numpy pandas scikit-learn matplotlib seaborn scipy h5py tables openpyxl xlrd
    )
) else (
    echo requirements.txt no encontrado, instalando dependencias basicas...
    python -m pip install numpy pandas scikit-learn matplotlib seaborn scipy h5py tables openpyxl xlrd
)

echo.
echo Configuracion del entorno completada.
echo Para activar manualmente en el futuro: call Regression_cocoa\Scripts\activate.bat
echo.

echo ============================================
echo CREACION Y NORMALIZACION DE DATASETS
echo ============================================
echo.

echo Ejecutando download_cocoa_dataset...
python data\create_dataset\download_cocoa_dataset.py

echo Ejecutando create_NIR2025_dataset...
python data\create_dataset\create_NIR2025_dataset.py

echo Ejecutando create_VIS2025_dataset...
python data\create_dataset\create_VIS2025_dataset.py

echo Ejecutando create_TEST_NIR2025_dataset...
python data\create_dataset\create_TEST_NIR2025_dataset.py

echo Ejecutando create_TEST_VIS2025_dataset...
python data\create_dataset\create_TEST_VIS2025_dataset.py

echo.
echo ============================================
echo NORMALIZANDO DATASETS CREADOS
echo ============================================
echo.

python data\data_normalization\data_normalize.py --auto

echo.
echo Datasets creados y normalizados exitosamente.
echo Archivos disponibles en la carpeta data/:
echo   - Datasets originales: train_*_cocoa_dataset.h5, test_*_cocoa_dataset.h5
echo   - Datasets normalizados: *_normalized.h5
echo.

echo ============================================
echo ENTRENAMIENTO INICIADO
echo ============================================

python Train.py --auto

echo ============================================
echo ENTRENAMIENTO COMPLETADO
echo ============================================

echo ============================================
echo TEST INICIADO
echo ============================================

python Test.py --auto

echo ============================================
echo TEST COMPLETADO
echo ============================================
