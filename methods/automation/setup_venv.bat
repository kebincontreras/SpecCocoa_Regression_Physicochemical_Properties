@echo off
REM =============================================================================
REM Virtual Environment Management
REM =============================================================================

setlocal enabledelayedexpansion

set ENV_NAME=%1
if "%ENV_NAME%"=="" set ENV_NAME=Regressio_cocoa_venv

echo ============================================
echo   Virtual Environment Management
echo ============================================

set NEED_NEW_ENV=0
set NEED_INSTALL_PACKAGES=0

echo Checking virtual environment: %ENV_NAME%

REM Step 1: Check if environment exists and is valid
if exist "%ENV_NAME%\Scripts\python.exe" (
    echo Virtual environment found: %ENV_NAME%
    echo Checking if environment is functional...
    
    REM Test if the environment can be activated and Python works
    "%ENV_NAME%\Scripts\python.exe" -c "import sys; print('Python version:', sys.version_info)" >nul 2>&1
    if %errorlevel% equ 0 (
        echo Environment is functional, will preserve and update packages only
        set NEED_NEW_ENV=0
        set NEED_INSTALL_PACKAGES=1
    ) else (
        echo  Environment appears corrupted - will recreate
        echo Removing corrupted environment...
        rmdir /s /q "%ENV_NAME%" >nul 2>&1
        set NEED_NEW_ENV=1
        set NEED_INSTALL_PACKAGES=1
    )
) else (
    echo 🔄 Environment not found - will create new one
    set NEED_NEW_ENV=1
    set NEED_INSTALL_PACKAGES=1
)

REM Step 2: Create environment if needed
if %NEED_NEW_ENV%==1 (
    echo.
    echo Creating new virtual environment...
    
    REM Remove existing broken environment
    if exist "%ENV_NAME%" (
        echo Removing broken environment...
        rmdir /s /q "%ENV_NAME%" >nul 2>&1
    )
    
    REM Create new environment
    python -m venv "%ENV_NAME%"
    if %errorlevel% neq 0 (
        echo  Failed to create environment with venv, trying alternative...
        python -m venv "%ENV_NAME%" --without-pip
        if %errorlevel% neq 0 (
            echo  Environment creation failed completely
            echo.
            echo SOLUTION: Check your Python installation
            pause
            exit /b 1
        )
    )

    if exist "%ENV_NAME%\Scripts\python.exe" (
        echo Virtual environment created, checking pip...
        "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo  pip not found in new environment, attempting to install with get-pip.py...
            if exist "get-pip.py" del "get-pip.py"
            curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
            if exist "get-pip.py" (
                "%ENV_NAME%\Scripts\python.exe" get-pip.py >nul 2>&1
                del "get-pip.py" >nul 2>&1
            )
        )
        "%ENV_NAME%\Scripts\python.exe" -m pip --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo  Error: Could not install pip in the new environment. Please check your Python installation.
            echo  Try reinstalling Python from python.org and ensure pip is included.
            pause
            exit /b 1
        )
        echo Virtual environment created and pip is available
        set NEED_INSTALL_PACKAGES=1
    ) else (
        echo  Environment creation verification failed
        echo.
        echo SOLUTION: Check your Python installation
        pause
        exit /b 1
    )
)

REM Step 3: Activate environment
echo.
if %NEED_NEW_ENV%==0 (
    echo Using existing virtual environment...
) else (
    echo New environment created successfully
)
echo Activating virtual environment...
call "%ENV_NAME%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Failed to activate environment
    echo.
    pause
    exit /b 1
)
echo Environment activated successfully

echo Virtual environment setup completed successfully.
echo Note: Existing environment preserved, packages will be updated as needed.
exit /b 0
