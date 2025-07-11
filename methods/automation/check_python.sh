#!/bin/bash
# =============================================================================
# Python Environment Check and Setup for Linux
# =============================================================================

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "============================================"
echo "   Python Environment Check"
echo "============================================"

if ! command_exists python3; then
    echo " Error: Python3 is not installed or not in PATH."
    echo ""
    echo "SOLUTIONS:"
    echo "1. Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "2. CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "3. Fedora: sudo dnf install python3 python3-pip"
    echo "4. Arch Linux: sudo pacman -S python python-pip"
    echo "5. macOS: brew install python3"
    echo ""
    exit 1
fi

echo "Found Python version:"
python3 --version

# Get detailed Python info for troubleshooting
PYTHON_PATH=$(python3 -c "import sys; print(sys.executable)")
echo "Python executable: $PYTHON_PATH"

# Check if this is a problematic Python installation (like some conda installations)
if echo "$PYTHON_PATH" | grep -qi "conda\|anaconda"; then
    echo ""
    echo "  WARNING: Detected Conda/Anaconda Python"
    echo "This may cause virtual environment issues."
    echo ""
    echo "RECOMMENDATION:"
    echo "1. Use 'conda create' instead of venv, OR"
    echo "2. Install standalone Python from your package manager"
    echo ""
    echo "Continuing anyway..."
    sleep 3
fi

# Check Python architecture
PYTHON_ARCH=$(python3 -c "import platform; print(platform.architecture()[0])")
echo "Python architecture: $PYTHON_ARCH"

# Check and fix pip if needed
python3 -m pip --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Warning: pip not available. Auto-fixing..."
    bash methods/automation/fix_pip.sh
    python3 -m pip --version >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo " Error: Could not fix pip. Please install pip manually."
        echo "Try: sudo apt install python3-pip (Ubuntu/Debian)"
        exit 1
    fi
fi

# Check venv module
python3 -m venv --help >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo " Error: venv module not available."
    echo "Install with: sudo apt install python3-venv (Ubuntu/Debian)"
    exit 1
fi

echo " Python environment check completed successfully."
exit 0
