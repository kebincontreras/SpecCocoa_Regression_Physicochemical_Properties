@echo off
REM =============================================================================
REM Training and Testing Automation
REM =============================================================================

setlocal enabledelayedexpansion

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=Regressio_cocoa_venv

echo ============================================
echo   Executing Training and Test
echo ============================================

REM Configure environment variables to suppress warnings
call methods\automation\configure_warnings.bat

REM Activate environment
call "%ENV_NAME%\Scripts\activate.bat"

REM Configure specific suppression of pkg_resources warnings
set PYTHONWARNINGS=ignore::UserWarning:lightning_utilities

echo Executing train.py...
python -W ignore::UserWarning:lightning_utilities train.py
if %errorlevel% neq 0 (
    echo ERROR: Failed in train.py
    exit /b 1
)

echo Executing test_industrial.py...
python -W ignore::UserWarning:lightning_utilities test_industrial.py
if %errorlevel% neq 0 (
    echo ERROR: Failed in test_industrial.py
    exit /b 1
)

deactivate
echo Training and testing completed successfully.
exit /b 0