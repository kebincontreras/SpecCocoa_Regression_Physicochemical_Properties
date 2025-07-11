@echo off
REM =============================================================================
REM Fix pip installation
REM =============================================================================

echo Attempting to fix pip installation...
python -m ensurepip --upgrade --default-pip --user >nul 2>&1
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Downloading and installing pip...
    if exist "get-pip.py" del "get-pip.py"
    curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >nul 2>&1 || powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'" >nul 2>&1
    if exist "get-pip.py" (
        python get-pip.py --user >nul 2>&1
        del "get-pip.py" >nul 2>&1
    )
)
exit /b 0
