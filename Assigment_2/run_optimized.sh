#!/bin/bash

# Runtime optimization script for H100

# Check if array size is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <array_size>"
    echo "Example: $0 134217728"
    exit 1
fi

ARRAY_SIZE=$1

echo "Setting up H100 optimizations..."

# 1. Set GPU to maximum performance mode (if you have permissions)
echo "Attempting to set GPU to max performance..."
nvidia-smi -pm 1 2>/dev/null || echo "Note: Could not set persistence mode (requires sudo)"
nvidia-smi -ac 2619,1980 2>/dev/null || echo "Note: Could not set application clocks (requires sudo)"

# 2. Set CUDA environment variables for better performance
export CUDA_LAUNCH_BLOCKING=0  # Enable asynchronous kernel launches
export CUDA_DEVICE_MAX_CONNECTIONS=32  # Increase concurrent kernel limit
export CUDA_FORCE_PTX_JIT=0  # Disable PTX JIT compilation

# 3. Show current GPU status
echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=name,persistence_mode,clocks.gr,clocks.mem,temperature.gpu,power.draw --format=csv,noheader,nounits

# 4. Compile with optimizations if needed
if [ ! -f kernel_optimized ] || [ kernel_optimized.cu -nt kernel_optimized ]; then
    echo ""
    echo "Recompiling with optimizations..."
    bash compile_optimized.sh
fi

# 5. Run the optimized kernel
echo ""
echo "Running optimized kernel with array size: $ARRAY_SIZE"
echo "=========================================="
./kernel_optimized $ARRAY_SIZE

# 6. For comparison, you can run multiple times and take the best
echo ""
echo "Running 3 more times to find best performance..."
echo "=========================================="
for i in {2..4}; do
    echo "Run $i:"
    ./kernel_optimized $ARRAY_SIZE | grep -E "GPU Sort Speed|H2D Transfer|Kernel Time|D2H Transfer"
    echo ""
done
