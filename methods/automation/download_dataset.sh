#!/bin/bash
# =============================================================================
# Dataset Download Automation Script for Linux
# =============================================================================

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "============================================"
echo "   Dataset Download Process"
echo "============================================"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models figures data data/raw_dataset logs

echo "Downloading Spectral Signatures of Cocoa Beans Dataset..."

if [ -f "data/raw_dataset/spectral-signatures-cocoa-beans.rar" ]; then
    echo " Dataset archive already exists. Skipping download..."
    exit 0
else
    echo "⬇  Starting download from HuggingFace..."
    if command_exists curl; then
        curl -L -o "data/raw_dataset/spectral-signatures-cocoa-beans.rar" "https://huggingface.co/datasets/kebincontreras/Spectral_signatures_of_cocoa_beans/resolve/main/Spectral_signatures_of_cocoa_beans.rar"
    elif command_exists wget; then
        wget -O "data/raw_dataset/spectral-signatures-cocoa-beans.rar" "https://huggingface.co/datasets/kebincontreras/Spectral_signatures_of_cocoa_beans/resolve/main/Spectral_signatures_of_cocoa_beans.rar"
    else
        echo " Error: Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo " Error: Failed to download dataset"
        exit 1
    fi
    echo " Dataset download completed successfully!"
    exit 0
fi
