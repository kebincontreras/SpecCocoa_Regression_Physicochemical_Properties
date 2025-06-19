@echo off
echo Creando entorno virtual...
python -m venv Regression_cocoa

echo Activando entorno virtual...
call Regression_cocoa\Scripts\activate.bat

echo Instalando librerias desde requirements.txt...
pip install -r requirements.txt

echo Configuracion completada!
echo Para activar el entorno virtual en el futuro, ejecute: Regression_cocoa\Scripts\activate.bat
pause
pip install jupyter
pip install openpyxl
pip install tabulate
pip install tensorflow

echo Configuracion completada!
echo Para activar el entorno virtual en el futuro, ejecute: Regression_cocoa\Scripts\activate.bat
pause
