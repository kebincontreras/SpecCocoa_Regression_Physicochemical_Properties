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
