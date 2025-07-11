@echo off
REM =============================================================================
REM Dataset Extraction Automation Script
REM =============================================================================

setlocal enabledelayedexpansion

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=Regressio_cocoa_venv

REM Configuration
set DATASET_DIR=data\raw_dataset\Spectral_signatures_of_cocoa_beans
set KEY_FILE=%DATASET_DIR%\Labels.xlsx
set RAR_FILE=data\raw_dataset\spectral-signatures-cocoa-beans.rar

echo ============================================
echo   Dataset Extraction Process
echo ============================================

if exist "%KEY_FILE%" (
    echo Dataset already extracted and verified. Skipping extraction...
    exit /b 0
)

echo  Extracting dataset...
set EXTRACTION_SUCCESS=0

REM Method 1: Try 7-Zip first (most common on Windows)
echo Trying 7-Zip extraction...
    
    REM Check for 7-Zip in common locations
    set SEVENZIP_FOUND=0
    if exist "C:\Program Files\7-Zip\7z.exe" (
        set SEVENZIP_PATH="C:\Program Files\7-Zip\7z.exe"
        set SEVENZIP_FOUND=1
    ) else if exist "C:\Program Files (x86)\7-Zip\7z.exe" (
        set SEVENZIP_PATH="C:\Program Files (x86)\7-Zip\7z.exe"
        set SEVENZIP_FOUND=1
    ) else (
        where 7z >nul 2>&1
        if %errorlevel% equ 0 (
            set SEVENZIP_PATH=7z
            set SEVENZIP_FOUND=1
        )
    )
    
    if %SEVENZIP_FOUND%==1 (
        echo Found 7-Zip, attempting extraction...
        %SEVENZIP_PATH% x "%RAR_FILE%" -o"data\raw_dataset" -y >nul 2>&1
        if exist "%KEY_FILE%" (
            echo  Dataset extracted successfully with 7-Zip!
            set EXTRACTION_SUCCESS=1
            goto :extraction_complete
        ) else (
            echo   7-Zip extraction failed, trying next method...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    ) else (
        echo 7-Zip not found, trying WinRAR...
    )
    
    REM Method 2: Try WinRAR
    echo Trying WinRAR extraction...
    
    REM Check for WinRAR in common locations
    set WINRAR_FOUND=0
    if exist "C:\Program Files\WinRAR\WinRAR.exe" (
        set WINRAR_PATH="C:\Program Files\WinRAR\WinRAR.exe"
        set WINRAR_FOUND=1
    ) else if exist "C:\Program Files (x86)\WinRAR\WinRAR.exe" (
        set WINRAR_PATH="C:\Program Files (x86)\WinRAR\WinRAR.exe"
        set WINRAR_FOUND=1
    ) else if exist "C:\Program Files\WinRAR\unrar.exe" (
        set WINRAR_PATH="C:\Program Files\WinRAR\unrar.exe"
        set WINRAR_FOUND=1
    ) else if exist "C:\Program Files (x86)\WinRAR\unrar.exe" (
        set WINRAR_PATH="C:\Program Files (x86)\WinRAR\unrar.exe"
        set WINRAR_FOUND=1
    ) else (
        where winrar >nul 2>&1
        if %errorlevel% equ 0 (
            set WINRAR_PATH=winrar
            set WINRAR_FOUND=1
        ) else (
            where unrar >nul 2>&1
            if %errorlevel% equ 0 (
                set WINRAR_PATH=unrar
                set WINRAR_FOUND=1
            )
        )
    )
    
    if %WINRAR_FOUND%==1 (
        echo Found WinRAR/unrar, attempting extraction...
        %WINRAR_PATH% x "%RAR_FILE%" "data\raw_dataset\" >nul 2>&1
        if exist "%KEY_FILE%" (
            echo  Dataset extracted successfully with WinRAR!
            set EXTRACTION_SUCCESS=1
            goto :extraction_complete
        ) else (
            echo   WinRAR extraction failed, trying Python packages...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    ) else (
        echo WinRAR not found, trying Python packages...
    )
    
    REM Method 3: Try Python packages for extraction
    echo Trying Python-based extraction packages...
    
    REM Try rarfile package
    "%ENV_NAME%\Scripts\python.exe" -c "import rarfile" >nul 2>&1
    if %errorlevel% neq 0 (
        echo Installing rarfile package...
        pip install rarfile --quiet >nul 2>&1
    )
    
    echo Testing rarfile extraction...
    "%ENV_NAME%\Scripts\python.exe" -c "import rarfile; rf = rarfile.RarFile('data\\raw_dataset\\spectral-signatures-cocoa-beans.rar'); rf.extractall('data\\raw_dataset'); rf.close(); print('Python rarfile extraction completed')" 2>nul
    if %errorlevel% equ 0 (
        REM Verify extraction worked by checking for key file
        if exist "%KEY_FILE%" (
            echo  Dataset extracted successfully with Python rarfile!
            set EXTRACTION_SUCCESS=1
            goto :extraction_complete
        ) else (
            echo   rarfile extraction created directory but no files found. Trying patoolib...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    ) else (
        echo   Python rarfile extraction failed (missing unrar tool), trying patoolib...
        if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        
        REM Try patoolib package
        "%ENV_NAME%\Scripts\python.exe" -c "import patoolib" >nul 2>&1
        if %errorlevel% neq 0 (
            echo Installing patoolib package...
            pip install patoolib --quiet >nul 2>&1
        )
        
        echo Testing patoolib extraction...
        "%ENV_NAME%\Scripts\python.exe" -c "import patoolib; patoolib.extract_archive('data\\raw_dataset\\spectral-signatures-cocoa-beans.rar', outdir='data\\raw_dataset'); print('Python patoolib extraction completed')" 2>nul
        if %errorlevel% equ 0 (
            REM Verify extraction worked by checking for key file
            if exist "%KEY_FILE%" (
                echo  Dataset extracted successfully with Python patoolib!
                set EXTRACTION_SUCCESS=1
                goto :extraction_complete
            ) else (
                echo   patoolib extraction created directory but no files found. Trying PowerShell...
                if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
            )
        ) else (
            echo   Python patoolib extraction failed, trying PowerShell...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
    )
    
    REM Method 4: Try PowerShell as last resort
    echo Trying PowerShell method (rename to .zip)...
    copy "%RAR_FILE%" "data\raw_dataset\dataset_temp.zip" >nul 2>&1
    if exist "data\raw_dataset\dataset_temp.zip" (
        powershell -Command "try { Expand-Archive -Path 'data\raw_dataset\dataset_temp.zip' -DestinationPath 'data\raw_dataset\' -Force; exit 0 } catch { exit 1 }" >nul 2>&1
        if exist "%KEY_FILE%" (
            echo  Dataset extracted successfully with PowerShell!
            set EXTRACTION_SUCCESS=1
            del "data\raw_dataset\dataset_temp.zip" >nul 2>&1
            goto :extraction_complete
        ) else (
            echo   PowerShell extraction failed...
            if exist "%DATASET_DIR%" rmdir /s /q "%DATASET_DIR%" >nul 2>&1
        )
        del "data\raw_dataset\dataset_temp.zip" >nul 2>&1
    )
    
    REM All methods failed
    echo.
    echo ============================================
    echo   Manual extraction required
    echo ============================================
    echo  All automatic extraction methods failed.
    echo.
    echo The issue: Python packages like rarfile need unrar.exe tool
    echo to extract RAR files, but it's not installed on your system.
    echo.
    echo SOLUTION OPTIONS:
    echo.
    echo Option 1 - Install 7-Zip (Recommended):
    echo   1. Download from: https://www.7-zip.org/
    echo   2. Install and add to PATH
    echo   3. Run this script again
    echo.
    echo Option 2 - Manual extraction:
    echo   1. Right-click: data\raw_dataset\spectral-signatures-cocoa-beans.rar
    echo   2. Choose "Extract All" or use any extraction tool
    echo   3. Extract to: data\raw_dataset\ folder
    echo.
    echo Option 3 - Install WinRAR:
    echo   1. Download from: https://www.winrar.es/
    echo   2. Install and add to PATH
    echo   3. Run this script again
    echo.
    echo After extraction, you should have:
    echo   data\raw_dataset\Spectral_signatures_of_cocoa_beans\Labels.xlsx
    echo   data\raw_dataset\Spectral_signatures_of_cocoa_beans\02_07_2024\
    echo   data\raw_dataset\Spectral_signatures_of_cocoa_beans\09_05_2024\
    echo.
    echo Press any key to continue with simulated data...
    pause >nul
    exit /b 1
    
    :extraction_complete
    if exist "%KEY_FILE%" (
        echo  Dataset extraction verified - Labels.xlsx found!
        
        REM Check for main directories to confirm extraction quality
        if exist "%DATASET_DIR%\02_07_2024" (
            echo  Date directory 02_07_2024 found
        )
        if exist "%DATASET_DIR%\09_05_2024" (
            echo  Date directory 09_05_2024 found
        )
        
        echo  Dataset extraction completed successfully!
        exit /b 0
    ) else (
        echo   Warning: Extraction verification failed - Labels.xlsx not found.
        exit /b 1
    )
