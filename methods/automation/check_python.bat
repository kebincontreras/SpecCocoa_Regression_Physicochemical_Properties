@echo off
REM =============================================================================
REM Python Environment Check and Setup
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================
echo   Python Environment Check
echo ============================================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Error: Python is not installed or not in PATH.
    echo.
    echo SOLUTIONS:
    echo 1. Install Python from: https://python.org/downloads/
    echo 2. Make sure to check "Add Python to PATH" during installation
    echo 3. Restart Command Prompt after installation
    echo.
    pause
    exit /b 1
)

echo Found Python version:
python --version

REM Get detailed Python info for troubleshooting
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_PATH=%%i
echo Python executable: %PYTHON_PATH%

REM Check Python architecture
for /f "tokens=*" %%i in ('python -c "import platform; print(platform.architecture()[0])"') do set PYTHON_ARCH=%%i
echo Python architecture: %PYTHON_ARCH%

echo Python environment check completed successfully.
exit /b 0
    echo.
    echo RECOMMENDATION:
    echo 1. Use 'conda create' instead of venv, OR
    echo 2. Install standalone Python from python.org
    echo.
    echo Continuing anyway...
    timeout /t 3 /nobreak >nul
)

REM Check Python architecture
for /f "tokens=*" %%i in ('python -c "import platform; print(platform.architecture()[0])"') do set PYTHON_ARCH=%%i
echo Python architecture: %PYTHON_ARCH%

REM Check and fix pip if needed
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: pip not available. Auto-fixing...
    call methods\automation\fix_pip.bat
    python -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Could not fix pip. Please reinstall Python with pip included.
        pause
        exit /b 1
    )
)

REM Check venv module
python -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: venv module not available. Please reinstall Python.
    pause
    exit /b 1
)

echo Python environment check completed successfully.
exit /b 0
