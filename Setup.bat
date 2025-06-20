@echo off
chcp 65001 >nul

@echo off
echo ============================================
echo     CONFIGURACION DEL ENTORNO PYTHON
echo ============================================
echo.

echo 🔧 Creando entorno virtual...
python -m venv Regression_cocoa

echo 🔄 Activando entorno virtual...
call Regression_cocoa\Scripts\activate.bat

echo 📦 Instalando dependencias desde requirements.txt...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo ⚠️ Error instalando desde requirements.txt, intentando instalación básica...
        python -m pip install numpy pandas scikit-learn matplotlib seaborn scipy h5py tables openpyxl xlrd
    )
) else (
    echo ⚠️ requirements.txt no encontrado, instalando dependencias básicas...
    python -m pip install numpy pandas scikit-learn matplotlib seaborn scipy h5py tables openpyxl xlrd
)

echo.
echo ✅ Configuracion del entorno completada!
echo 💡 Para activar el entorno virtual en el futuro, ejecute: Regression_cocoa\Scripts\activate.bat
echo.
  
  

echo ============================================
echo      CREACION Y NORMALIZACION DE DATASETS
echo ============================================
echo.

REM Verificar y activar entorno virtual
if exist "Regression_cocoa\Scripts\activate.bat" (
    echo 🔄 Activando entorno virtual...
    call Regression_cocoa\Scripts\activate.bat
) else (
    echo ❌ Entorno virtual no encontrado
    echo 💡 Ejecute setup_environment.bat primero
    exit /b 1
)

echo Ejecutando download_cocoa_dataset...
python data\create_dataset\download_cocoa_dataset.py

echo.
echo 🚫 Deshabilitando guardado de archivos PKL...
python remove_pkl_generation.py
echo.

echo 🏗️ Ejecutando scripts de creacion de datasets...
echo.

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
echo    NORMALIZANDO DATASETS CREADOS
echo ============================================
echo.

echo.
echo Ejecutando normalizacion de datasets...
python data\data_normalization\data_normalize.py --auto

echo.
echo Verificando archivos normalizados...
set "normalized_missing=0"
if not exist "data\train_NIR_cocoa_dataset_normalized.h5" set /a normalized_missing+=1
if not exist "data\train_VIS_cocoa_dataset_normalized.h5" set /a normalized_missing+=1
if not exist "data\test_NIR_cocoa_dataset_normalized.h5" set /a normalized_missing+=1
if not exist "data\test_VIS_cocoa_dataset_normalized.h5" set /a normalized_missing+=1

echo.
echo ============================================
echo         PROCESO COMPLETADO
echo ============================================

echo.
echo 🎉 Datasets creados y normalizados exitosamente!
echo.
echo 📂 Archivos disponibles en la carpeta data/:
echo    • Datasets originales: train_*_cocoa_dataset.h5, test_*_cocoa_dataset.h5
echo    • Datasets normalizados: *_normalized.h5
echo.

REM Script termina automáticamente sin pause
echo ============================================
echo         PROCESO COMPLETADO
echo ============================================
echo.

python data\Train.py --auto

echo ============================================
echo        ENTRENAMIENTO COMPLETADO
echo ============================================
echo.

python data\Train.py --auto


echo ============================================
echo        TEST COMPLETADO
echo ============================================
echo.