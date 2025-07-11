#!/bin/bash
# =============================================================================
# Environment Optimization for Linux/macOS
# =============================================================================

# Configure total warning suppression
export PYTHONWARNINGS="ignore"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_QUIET=1

# Configure environment variables for maximum performance
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Configure CUDA if available
if command -v nvidia-smi >/dev/null 2>&1; then
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_LAUNCH_BLOCKING=0
fi

# Configure OpenMP for better CPU performance
export OMP_NUM_THREADS=$(nproc)

# Configure memory limits if needed
ulimit -v unlimited 2>/dev/null || true

echo "Environment optimized for maximum performance and minimal warnings"
