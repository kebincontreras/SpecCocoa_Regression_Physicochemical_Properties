#!/bin/bash
# =============================================================================
# Training and Testing Automation for Linux
# =============================================================================

ENV_NAME=${1:-"Regressio_cocoa_venv"}

echo "============================================"
echo "   Executing Training and Test"
echo "============================================"

# Activate environment
source "${ENV_NAME}/bin/activate"

# Configure warning suppression at system level
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Apply system optimizations
echo "Applying system optimizations..."
source methods/automation/optimize_environment.sh

echo "Executing train.py..."
python -W ignore train.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed in train.py"
    exit 1
fi

echo "Executing test_industrial.py..."
python -W ignore test_industrial.py
if [ $? -ne 0 ]; then
    echo "ERROR: Failed in test_industrial.py"
    exit 1
fi
fi

echo "Training and testing completed successfully."
exit 0
