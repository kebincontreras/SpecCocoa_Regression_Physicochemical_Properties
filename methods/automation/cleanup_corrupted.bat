@echo off
REM =============================================================================
REM Cleanup Corrupted Python Distributions - Dynamic Version
REM =============================================================================

echo ============================================
echo   Limpiando Distribuciones Corruptas
echo ============================================

REM Get Python site-packages path dynamically
for /f "tokens=*" %%i in ('python -c "import site; print(site.getsitepackages()[0])"') do set SITE_PACKAGES=%%i

echo Buscando distribuciones corruptas en: %SITE_PACKAGES%

REM Method 1: Direct folder deletion with for loop
echo Method 1: Direct deletion...
for /d %%i in ("%SITE_PACKAGES%\~*") do (
    echo Eliminando: %%i
    rmdir /s /q "%%i" 2>nul
)

REM Method 2: Specific problematic packages
echo Method 2: Specific problematic packages...
if exist "%SITE_PACKAGES%\~inja2" (
    echo Removing ~inja2 specifically...
    rmdir /s /q "%SITE_PACKAGES%\~inja2" 2>nul
)
if exist "%SITE_PACKAGES%\~ympy" (
    echo Removing ~ympy specifically...
    rmdir /s /q "%SITE_PACKAGES%\~ympy" 2>nul
)

REM Method 3: Reinstall problematic packages cleanly
echo Method 3: Clean reinstallation...
set PYTHONWARNINGS=ignore
set PIP_DISABLE_PIP_VERSION_CHECK=1
pip uninstall -y jinja2 sympy --quiet 2>nul
pip install jinja2 sympy --quiet --disable-pip-version-check 2>nul

echo Cleanup of corrupted distributions completed.
echo.
