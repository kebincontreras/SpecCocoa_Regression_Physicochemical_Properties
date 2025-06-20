@echo off
chcp 65001 >nul

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

echo Verificando que existan los datasets originales...
set "missing_files=0"
if not exist "data\train_NIR_cocoa_dataset.h5" set /a missing_files+=1
if not exist "data\train_VIS_cocoa_dataset.h5" set /a missing_files+=1
if not exist "data\test_NIR_cocoa_dataset.h5" set /a missing_files+=1
if not exist "data\test_VIS_cocoa_dataset.h5" set /a missing_files+=1

if %missing_files% gtr 0 (
    echo ⚠️ Faltan algunos datasets originales. Verificar errores.
) else (
    echo ✅ Todos los datasets originales fueron creados correctamente.
)

echo.
echo Ejecutando normalizacion de datasets...
python data\create_dataset\normalize_datasets.py

echo.
echo Verificando archivos normalizados...
set "normalized_missing=0"
if not exist "data\train_NIR_cocoa_dataset_normalized.h5" set /a normalized_missing+=1
if not exist "data\train_VIS_cocoa_dataset_normalized.h5" set /a normalized_missing+=1
if not exist "data\test_NIR_cocoa_dataset_normalized.h5" set /a normalized_missing+=1
if not exist "data\test_VIS_cocoa_dataset_normalized.h5" set /a normalized_missing+=1

if %normalized_missing% gtr 0 (
    echo ⚠️ Algunos datasets normalizados no se generaron correctamente.
    echo    Ejecute manualmente: python data\create_dataset\normalize_datasets.py
) else (
    echo ✅ Todos los datasets fueron normalizados correctamente.
)

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

REM Verificar archivos creados
echo 📁 Verificando archivos creados...
if exist "data\train_NIR_cocoa_dataset.h5" (
    echo ✅ train_NIR_cocoa_dataset.h5
) else (
    echo ❌ train_NIR_cocoa_dataset.h5 no encontrado
)

if exist "data\train_VIS_cocoa_dataset.h5" (
    echo ✅ train_VIS_cocoa_dataset.h5
) else (
    echo ❌ train_VIS_cocoa_dataset.h5 no encontrado
)

echo.
echo 🎉 Todos los datasets han sido procesados!
echo.
echo 📂 Archivos disponibles en la carpeta data/:
echo    • Datasets originales: train_*_cocoa_dataset.h5, test_*_cocoa_dataset.h5
echo    • Datasets normalizados: *_normalized.h5
echo.
