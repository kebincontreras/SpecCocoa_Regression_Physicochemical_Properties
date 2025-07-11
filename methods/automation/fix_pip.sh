#!/bin/bash
# =============================================================================
# Fix pip installation for Linux
# =============================================================================

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Attempting to fix pip installation..."
python3 -m ensurepip --upgrade --default-pip --user >/dev/null 2>&1 || true
python3 -m pip --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Downloading and installing pip..."
    if [ -f "get-pip.py" ]; then
        rm "get-pip.py"
    fi
    if command_exists curl; then
        curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1
    elif command_exists wget; then
        wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py >/dev/null 2>&1
    fi
    if [ -f "get-pip.py" ]; then
        python3 get-pip.py --user >/dev/null 2>&1
        rm "get-pip.py" >/dev/null 2>&1
    fi
fi
exit 0
