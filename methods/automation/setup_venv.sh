#!/bin/bash
# =============================================================================
# Virtual Environment Management for Linux
# =============================================================================

ENV_NAME=${1:-"Regressio_cocoa_venv"}

echo "============================================"
echo "   Virtual Environment Management"
echo "============================================"

NEED_NEW_ENV=0
NEED_INSTALL_PACKAGES=0

echo "Checking virtual environment: $ENV_NAME"

# Step 1: Check if environment exists and is valid
if [ -f "${ENV_NAME}/bin/python" ]; then
    echo "Virtual environment found: $ENV_NAME"
    echo "Checking if environment is functional..."
    
    # Test if the environment can be activated and Python works
    "${ENV_NAME}/bin/python" -c "import sys; print('Python version:', sys.version_info)" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Environment is functional, will preserve and update packages only"
        NEED_NEW_ENV=0
        NEED_INSTALL_PACKAGES=1
    else
        echo " Environment appears corrupted - will recreate"
        echo "Removing corrupted environment..."
        rm -rf "$ENV_NAME" >/dev/null 2>&1
        NEED_NEW_ENV=1
        NEED_INSTALL_PACKAGES=1
    fi
else
    echo "Environment not found - will create new one"
    NEED_NEW_ENV=1
    NEED_INSTALL_PACKAGES=1
fi

# Step 2: Create environment if needed
if [ $NEED_NEW_ENV -eq 1 ]; then
    echo ""
    echo "Creating new virtual environment..."
    
    # Remove existing broken environment
    if [ -d "$ENV_NAME" ]; then
        echo "Removing broken environment..."
        rm -rf "$ENV_NAME" >/dev/null 2>&1
    fi
    
    # Create new environment
    python3 -m venv "$ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Failed to create environment with venv"
        echo ""
        echo "SOLUTION: Check Python installation and try:"
        echo "sudo apt install python3-venv python3-dev"
        exit 1
    fi

    if [ -f "${ENV_NAME}/bin/python" ]; then
        echo "Virtual environment created, checking pip..."
        "${ENV_NAME}/bin/python" -m pip --version >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "🔧 pip not found in new environment, attempting to install..."
            if [ -f "get-pip.py" ]; then
                rm "get-pip.py"
            fi
            if command -v curl >/dev/null 2>&1; then
                curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1
            elif command -v wget >/dev/null 2>&1; then
                wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1
            fi
            if [ -f "get-pip.py" ]; then
                "${ENV_NAME}/bin/python" get-pip.py >/dev/null 2>&1
                rm "get-pip.py" >/dev/null 2>&1
            fi
        fi
        "${ENV_NAME}/bin/python" -m pip --version >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Error: Could not install pip in the new environment."
            echo "Please check your Python installation."
            exit 1
        fi
        echo "Virtual environment created and pip is available"
        NEED_INSTALL_PACKAGES=1
    else
        echo "Environment creation verification failed"
        exit 1
    fi
fi

# Step 3: Activate environment
echo ""
if [ $NEED_NEW_ENV -eq 0 ]; then
    echo "Using existing virtual environment..."
else
    echo "New environment created successfully"
fi
echo "Activating virtual environment..."
source "${ENV_NAME}/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to activate environment"
    exit 1
fi

# Configure warning suppression in the environment
export PYTHONWARNINGS="ignore"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "Environment activated successfully"

echo "Virtual environment setup completed successfully."
echo "Note: Existing environment preserved, packages will be updated as needed."
exit 0
