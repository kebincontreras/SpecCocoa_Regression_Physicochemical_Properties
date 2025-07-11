@echo off
REM =============================================================================
REM SpecCocoa Regression Project - Simplified Main Script
REM =============================================================================

setlocal enabledelayedexpansion

REM Project configuration
set PROJECT_NAME=SpecCocoa_Regression
set ENV_NAME=Regressio_cocoa_venv
set AUTOMATION_DIR=methods\automation

echo ============================================
echo   SpecCocoa Regression Project Setup and Run
echo ============================================
echo Starting automated setup process...
echo.

REM Step 1: Check Python Environment
echo [1/7] Checking Python Environment...
call "%AUTOMATION_DIR%\check_python.bat"
if %errorlevel% neq 0 (
    echo Python environment check failed.
    pause
    exit /b 1
)
echo.

REM Step 2: Setup Virtual Environment
echo [2/7] Setting up Virtual Environment...
call "%AUTOMATION_DIR%\setup_venv.bat" "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Virtual environment setup failed.
    pause
    exit /b 1
)
echo.                                                         

REM Step 3: Install Packages
echo [3/7] Installing Required Packages...
call "%AUTOMATION_DIR%\install_packages.bat" "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo Package installation failed.
    pause
    exit /b 1
)
echo.

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
