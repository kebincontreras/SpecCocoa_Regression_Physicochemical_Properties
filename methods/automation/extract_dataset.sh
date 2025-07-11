#!/bin/bash
# =============================================================================
# Dataset Extraction Automation Script for Linux
# =============================================================================

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

ENV_NAME=${1:-"Regressio_cocoa_venv"}

# Configuration
DATASET_DIR="data/raw_dataset/Spectral_signatures_of_cocoa_beans"
KEY_FILE="${DATASET_DIR}/Labels.xlsx"
RAR_FILE="data/raw_dataset/spectral-signatures-cocoa-beans.rar"

echo "============================================"
echo "   Dataset Extraction Process"
echo "============================================"

if [ -f "$KEY_FILE" ]; then
    echo "Dataset already extracted and verified. Skipping extraction..."
    exit 0
fi

echo "Extracting dataset..."
EXTRACTION_SUCCESS=0

# Activate environment
source "${ENV_NAME}/bin/activate"

# Method 1: Try unrar first (most common on Linux)
echo "Trying unrar extraction..."
if command_exists unrar; then
    echo "Found unrar, attempting extraction..."
    unrar x "$RAR_FILE" "data/raw_dataset/" >/dev/null 2>&1
    if [ -f "$KEY_FILE" ]; then
        echo "Dataset extracted successfully with unrar!"
        EXTRACTION_SUCCESS=1
    else
        echo "unrar extraction failed, trying next method..."
        rm -rf "$DATASET_DIR" >/dev/null 2>&1
    fi
else
    echo "unrar not found, trying 7zip..."
fi

# Method 2: Try 7zip
if [ $EXTRACTION_SUCCESS -eq 0 ]; then
    echo "Trying 7zip extraction..."
    if command_exists 7z; then
        echo "Found 7z, attempting extraction..."
        7z x "$RAR_FILE" -o"data/raw_dataset" -y >/dev/null 2>&1
        if [ -f "$KEY_FILE" ]; then
            echo "Dataset extracted successfully with 7zip!"
            EXTRACTION_SUCCESS=1
        else
            echo "7zip extraction failed, trying Python packages..."
            rm -rf "$DATASET_DIR" >/dev/null 2>&1
        fi
    else
        echo "7z not found, trying Python packages..."
    fi
fi

# Method 3: Try Python packages for extraction
if [ $EXTRACTION_SUCCESS -eq 0 ]; then
    echo "Trying Python-based extraction packages..."
    
    # Try rarfile package
    python -c "import rarfile" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Installing rarfile package..."
        pip install rarfile --quiet >/dev/null 2>&1
    fi
    
    echo "Testing rarfile extraction..."
    python -c "import rarfile; rf = rarfile.RarFile('data/raw_dataset/spectral-signatures-cocoa-beans.rar'); rf.extractall('data/raw_dataset'); rf.close(); print('Python rarfile extraction completed')" >/dev/null 2>&1
    if [ $? -eq 0 ] && [ -f "$KEY_FILE" ]; then
        echo "Dataset extracted successfully with Python rarfile!"
        EXTRACTION_SUCCESS=1
    else
        echo "Python rarfile extraction failed, trying patoolib..."
        rm -rf "$DATASET_DIR" >/dev/null 2>&1
        
        # Try patoolib package
        python -c "import patoolib" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Installing patoolib package..."
            pip install patoolib --quiet >/dev/null 2>&1
        fi
        
        echo "Testing patoolib extraction..."
        python -c "import patoolib; patoolib.extract_archive('data/raw_dataset/spectral-signatures-cocoa-beans.rar', outdir='data/raw_dataset'); print('Python patoolib extraction completed')" >/dev/null 2>&1
        if [ $? -eq 0 ] && [ -f "$KEY_FILE" ]; then
            echo "Dataset extracted successfully with Python patoolib!"
            EXTRACTION_SUCCESS=1
        else
            echo "Python patoolib extraction failed..."
            rm -rf "$DATASET_DIR" >/dev/null 2>&1
        fi
    fi
fi

# All methods failed
if [ $EXTRACTION_SUCCESS -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "   Manual extraction required"
    echo "============================================"
    echo "All automatic extraction methods failed."
    echo ""
    echo "SOLUTION OPTIONS:"
    echo ""
    echo "Option 1 - Install unrar (Recommended):"
    echo "  Ubuntu/Debian: sudo apt install unrar"
    echo "  CentOS/RHEL: sudo yum install unrar"
    echo "  Fedora: sudo dnf install unrar"
    echo "  Arch Linux: sudo pacman -S unrar"
    echo "  macOS: brew install unrar"
    echo ""
    echo "Option 2 - Install 7zip:"
    echo "  Ubuntu/Debian: sudo apt install p7zip-full"
    echo "  CentOS/RHEL: sudo yum install p7zip"
    echo "  Fedora: sudo dnf install p7zip"
    echo "  Arch Linux: sudo pacman -S p7zip"
    echo "  macOS: brew install p7zip"
    echo ""
    echo "Option 3 - Manual extraction:"
    echo "  Extract: data/raw_dataset/spectral-signatures-cocoa-beans.rar"
    echo "  To: data/raw_dataset/ folder"
    echo ""
    echo "After extraction, you should have:"
    echo "  data/raw_dataset/Spectral_signatures_of_cocoa_beans/Labels.xlsx"
    echo "  data/raw_dataset/Spectral_signatures_of_cocoa_beans/02_07_2024/"
    echo "  data/raw_dataset/Spectral_signatures_of_cocoa_beans/09_05_2024/"
    echo ""
    echo "Press Enter to continue..."
    read
    exit 1
else
    echo "Dataset extraction verified - Labels.xlsx found!"
    
    # Check for main directories to confirm extraction quality
    if [ -d "${DATASET_DIR}/02_07_2024" ]; then
        echo "Date directory 02_07_2024 found"
    fi
    if [ -d "${DATASET_DIR}/09_05_2024" ]; then
        echo "Date directory 09_05_2024 found"
    fi
    echo "Dataset extraction completed successfully!"
    exit 0
fi
