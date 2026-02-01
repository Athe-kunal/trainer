#!/bin/bash
# Source this file before running training scripts to fix compilation issues
# Usage: source setup_env.sh

# Fix for CUDA/nvcc JIT compilation - ensure C headers are found
export CPATH=/usr/include:/usr/include/x86_64-linux-gnu:${CPATH:-}
export CPLUS_INCLUDE_PATH=/usr/include:/usr/include/x86_64-linux-gnu:${CPLUS_INCLUDE_PATH:-}
export C_INCLUDE_PATH=/usr/include:/usr/include/x86_64-linux-gnu:${C_INCLUDE_PATH:-}

echo "✓ Environment configured for CUDA JIT compilation"
echo "  CPATH: $CPATH"
echo "  CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"
echo "  C_INCLUDE_PATH: $C_INCLUDE_PATH"
