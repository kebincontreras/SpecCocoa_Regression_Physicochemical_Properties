@echo off
REM =============================================================================
REM Clean Setup - Delete existing environment and create fresh one
REM =============================================================================

echo ============================================
echo   SpecCocoa Clean Setup
echo ============================================
echo This will DELETE the existing virtual environment and create a fresh one.
echo This ensures all dependencies are installed cleanly without conflicts.
echo.

set /p confirm="Are you sure you want to continue? (y/N): "
if /i not "%confirm%"=="y" (
    echo Setup cancelled.
    exit /b 0
)

echo.
echo [1/2] Removing existing virtual environment...
if exist "Regressio_cocoa_venv" (
    echo Deleting Regressio_cocoa_venv...
    rmdir /s /q "Regressio_cocoa_venv"
    echo ✅ Old environment removed.
) else (
    echo No existing environment found.
)

echo.
echo [2/2] Running fresh setup...
call setup.bat

echo.
echo ✅ Clean setup completed!
echo The pkg_resources warning should now be resolved.
pause
