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
if not exist "models" mkdir models
if not exist "figures" mkdir figures
if not exist "data" mkdir data
if not exist "data\raw_dataset" mkdir data\raw_dataset
if not exist "logs" mkdir logs

echo Downloading Spectral Signatures of Cocoa Beans Dataset...

if exist "data\raw_dataset\spectral-signatures-cocoa-beans.rar" (
    echo Dataset archive already exists. Skipping download...
    exit /b 0
) else (
    echo ⬇  Starting download from HuggingFace...
    curl -L -o "data\raw_dataset\spectral-signatures-cocoa-beans.rar" "https://huggingface.co/datasets/kebincontreras/Spectral_signatures_of_cocoa_beans/resolve/main/Spectral_signatures_of_cocoa_beans.rar"
    
    if %errorlevel% neq 0 (
        echo  Error: Failed to download dataset
        exit /b 1
    )
    echo Dataset download completed successfully!
    exit /b 0
)
