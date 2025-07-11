@echo off
REM =============================================================================
REM Processing Scripts Automation - Smart Version
REM =============================================================================

setlocal enabledelayedexpansion

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=Regressio_cocoa_venv

echo ============================================
echo   Executing Data Processing Scripts
echo ============================================

REM Activate environment first
call "%ENV_NAME%\Scripts\activate.bat"

echo Package verification skipped - all dependencies managed by requirements.txt
echo If import errors occur, they will be reported by the individual scripts
echo.

REM Check if NIR datasets exist
echo Checking NIR datasets...
if exist "data\train_nir_cocoa_dataset.h5" (
    if exist "data\test_nir_cocoa_dataset.h5" (
        echo NIR datasets already exist. Skipping create_NIR2025_dataset.py...
    ) else (
        echo Missing NIR test dataset. Running create_NIR2025_dataset.py...
        echo This process may take several minutes processing the dataset...
        python data/create_dataset/create_NIR2025_dataset.py
        if %errorlevel% neq 0 (
            echo ERROR: Failed in create_NIR2025_dataset.py
            exit /b 1
        )
        echo NIR datasets generated successfully.
    )
) else (
    echo Missing NIR datasets. Running create_NIR2025_dataset.py...
    echo This process may take several minutes processing the dataset...
    python data/create_dataset/create_NIR2025_dataset.py
    if %errorlevel% neq 0 (
        echo ERROR: Failed in create_NIR2025_dataset.py
        exit /b 1
    )
    echo NIR datasets generated successfully.
)
echo.

REM Check if VIS datasets exist
echo Checking VIS datasets...
if exist "data\train_vis_cocoa_dataset.h5" (
    if exist "data\test_vis_cocoa_dataset.h5" (
        echo VIS datasets already exist. Skipping create_VIS2025_dataset.py...
    ) else (
        echo Missing VIS test dataset. Running create_VIS2025_dataset.py...
        echo Processing VIS dataset...
        python data/create_dataset/create_VIS2025_dataset.py
        if %errorlevel% neq 0 (
            echo ERROR: Failed in create_VIS2025_dataset.py
            exit /b 1
        )
        echo VIS datasets generated successfully.
    )
) else (
    echo Missing VIS datasets. Running create_VIS2025_dataset.py...
    echo Processing VIS dataset...
    python data/create_dataset/create_VIS2025_dataset.py
    if %errorlevel% neq 0 (
        echo ERROR: Failed in create_VIS2025_dataset.py
        exit /b 1
    )
    echo VIS datasets generated successfully.
)
echo.

REM Check if normalized datasets exist
echo Checking normalized datasets...
if exist "data\train_nir_cocoa_dataset_normalized.h5" (
    if exist "data\test_nir_cocoa_dataset_normalized.h5" (
        if exist "data\train_vis_cocoa_dataset_normalized.h5" (
            if exist "data\test_vis_cocoa_dataset_normalized.h5" (
                echo Normalized datasets already exist. Skipping normalize_datasets.py...
            ) else (
                echo Missing some normalized datasets. Running normalize_datasets.py...
                echo Normalizing all generated datasets...
                python data/create_dataset/normalize_datasets.py
                if %errorlevel% neq 0 (
                    echo ERROR: Failed in normalize_datasets.py
                    exit /b 1
                )
                echo Normalized datasets generated successfully.
            )
        ) else (
            echo Missing some normalized datasets. Running normalize_datasets.py...
            echo Normalizing all generated datasets...
            python data/create_dataset/normalize_datasets.py
            if %errorlevel% neq 0 (
                echo ERROR: Failed in normalize_datasets.py
                exit /b 1
            )
            echo Normalized datasets generated successfully.
        )
    ) else (
        echo Missing some normalized datasets. Running normalize_datasets.py...
        echo Normalizing all generated datasets...
        python data/create_dataset/normalize_datasets.py
        if %errorlevel% neq 0 (
            echo ERROR: Failed in normalize_datasets.py
            exit /b 1
        )
        echo Normalized datasets generated successfully.
    )
) else (
    echo Missing normalized datasets. Running normalize_datasets.py...
    echo Normalizing all generated datasets...
    python data/create_dataset/normalize_datasets.py
    if %errorlevel% neq 0 (
        echo ERROR: Failed in normalize_datasets.py
        exit /b 1
    )
    echo Normalized datasets generated successfully.
)

echo.
echo Data processing scripts completed successfully.
exit /b 0
