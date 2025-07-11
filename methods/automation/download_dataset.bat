@echo off
REM =============================================================================
REM Dataset Download Automation Script
REM =============================================================================

setlocal enabledelayedexpansion

echo ============================================
echo   Dataset Download Process
echo ============================================

REM Create necessary directories
echo Creating necessary directories...
if not exist "models" (
    mkdir models
    echo     → Created models\ directory
)
if not exist "data" (
    mkdir data
    echo     → Created data\ directory
)
if not exist "data\raw_dataset" (
    mkdir data\raw_dataset
    echo     → Created data\raw_dataset\ directory
)
echo Directory structure verified successfully.
echo.

echo Downloading Spectral Signatures of Cocoa Beans Dataset...

if exist "data\raw_dataset\spectral-signatures-cocoa-beans.rar" (
    echo Dataset archive already exists in data\raw_dataset\
    echo Skipping download...
    exit /b 0
) else (
    echo Starting download from HuggingFace repository...
    echo Target: data\raw_dataset\spectral-signatures-cocoa-beans.rar
    echo This may take a few minutes depending on your internet connection...
    echo.
    curl -L -o "data\raw_dataset\spectral-signatures-cocoa-beans.rar" "https://huggingface.co/datasets/kebincontreras/Spectral_signatures_of_cocoa_beans/resolve/main/Spectral_signatures_of_cocoa_beans.rar"
    
    if %errorlevel% neq 0 (
        echo.
        echo ERROR: Failed to download dataset
        echo.
        echo Possible causes:
        echo 1. No internet connection
        echo 2. HuggingFace repository unavailable
        echo 3. Insufficient disk space
        echo.
        echo Please check your connection and try again.
        exit /b 1
    )
    echo.
    echo Dataset download completed successfully!
    echo File saved to: data\raw_dataset\spectral-signatures-cocoa-beans.rar
    exit /b 0
)
