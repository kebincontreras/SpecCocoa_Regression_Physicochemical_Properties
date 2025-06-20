@echo off
echo Activando entorno virtual...
call ..\..\Regression_cocoa\Scripts\activate.bat

echo Ejecutando scripts de creacion de datasets...
echo.

echo Ejecutando create_NIR2025_dataset...
python create_NIR2025_dataset.py

echo Ejecutando create_TEST_NIR2025_dataset...
python create_TEST_NIR2025_dataset.py

echo Ejecutando create_TEST_VIS2025_dataset...
python create_TEST_VIS2025_dataset.py

echo Ejecutando create_VIS2025_dataset...
python create_VIS2025_dataset.py

echo.
echo Todos los datasets han sido creados exitosamente!
pause
