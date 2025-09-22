#!/bin/bash

# Compilation script with optimizations for NVIDIA H100 (SM 9.0)

echo "Compiling kernel_optimized.cu with H100 optimizations..."

# Compiler flags explanation:
# -arch=sm_90: Target H100 architecture (SM 9.0)
# -O3: Maximum optimization level
# -use_fast_math: Use faster, less precise math operations
# -Xptxas -v: Show PTX statistics (registers, shared memory usage)
# -Xcompiler -march=native: Optimize for host CPU
# -Xcompiler -O3: Host code optimization
# --expt-relaxed-constexpr: Enable extended constexpr support

nvcc -arch=sm_90 \
     -O3 \
     -use_fast_math \
     -Xptxas -v \
     -Xcompiler -march=native \
     -Xcompiler -O3 \
     --expt-relaxed-constexpr \
     -o kernel_optimized \
     kernel_optimized.cu

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo ""
    echo "Run with: ./kernel_optimized <array_size>"
    echo "Example: ./kernel_optimized 134217728"
else
    echo "Compilation failed!"
    exit 1
fi
