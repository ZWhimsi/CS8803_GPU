#!/usr/bin/env python3
"""
Performance measurement script for optimized CUDA bitonic sort
Tests memory throughput, occupancy, and MOPE/s
"""

import subprocess
import sys
import re
import time

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1

def measure_performance():
    """Measure performance metrics"""
    print("🚀 Testing Optimized CUDA Bitonic Sort Performance")
    print("=" * 60)
    
    # Test array sizes
    test_sizes = [1024, 10000, 100000, 1000000, 10000000]
    
    for size in test_sizes:
        print(f"\n📊 Testing array size: {size:,}")
        print("-" * 40)
        
        # Compile and run
        print("🔨 Compiling...")
        compile_cmd = "nvcc -o kernel_chunked kernel_chunked.cu"
        stdout, stderr, returncode = run_command(compile_cmd)
        if returncode != 0:
            print(f"❌ Compilation failed: {stderr}")
            continue
            
        # Run the kernel
        print("🏃 Running kernel...")
        run_cmd = f"./kernel_chunked {size}"
        stdout, stderr, returncode = run_command(run_cmd)
        
        if returncode != 0:
            print(f"❌ Execution failed: {stderr}")
            continue
            
        # Parse performance metrics
        print("📈 Parsing performance metrics...")
        
        # Extract MOPE/s
        mope_match = re.search(r'GPU Sort Speed: ([\d.]+) million elements per second', stdout)
        if mope_match:
            mope_s = float(mope_match.group(1))
            print(f"🎯 MOPE/s: {mope_s:.2f}")
        else:
            print("❌ Could not extract MOPE/s")
            
        # Extract kernel time
        kernel_match = re.search(r'Kernel Time \(ms\): ([\d.]+)', stdout)
        if kernel_match:
            kernel_time = float(kernel_match.group(1))
            print(f"⏱️  Kernel Time: {kernel_time:.2f} ms")
        else:
            print("❌ Could not extract kernel time")
            
        # Extract memory transfer time
        h2d_match = re.search(r'H2D Transfer Time \(ms\): ([\d.]+)', stdout)
        d2h_match = re.search(r'D2H Transfer Time \(ms\): ([\d.]+)', stdout)
        if h2d_match and d2h_match:
            h2d_time = float(h2d_match.group(1))
            d2h_time = float(d2h_match.group(1))
            total_transfer = h2d_time + d2h_time
            print(f"💾 Memory Transfer: {total_transfer:.2f} ms (H2D: {h2d_time:.2f}, D2H: {d2h_time:.2f})")
        else:
            print("❌ Could not extract memory transfer time")
            
        # Run ncu for detailed metrics
        print("🔍 Running ncu profiler...")
        ncu_cmd = f"ncu --metric gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed --metric sm__warps_active.avg.pct_of_peak_sustained_active --print-summary per-gpu ./kernel_chunked {size}"
        ncu_stdout, ncu_stderr, ncu_returncode = run_command(ncu_cmd)
        
        if ncu_returncode == 0:
            # Extract memory throughput
            throughput_match = re.search(r'gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed\s+(\d+\.\d+)', ncu_stdout)
            if throughput_match:
                throughput = float(throughput_match.group(1))
                print(f"📊 Memory Throughput: {throughput:.2f}%")
            else:
                print("❌ Could not extract memory throughput")
                
            # Extract occupancy
            occupancy_match = re.search(r'sm__warps_active\.avg\.pct_of_peak_sustained_active\s+(\d+\.\d+)', ncu_stdout)
            if occupancy_match:
                occupancy = float(occupancy_match.group(1))
                print(f"🎯 Achieved Occupancy: {occupancy:.2f}%")
            else:
                print("❌ Could not extract occupancy")
        else:
            print(f"❌ ncu failed: {ncu_stderr}")
            
        print("✅ Test completed")
        
        # Performance targets
        print("\n🎯 Performance Targets:")
        print(f"   MOPE/s: {mope_s:.2f} / 900 (target)")
        print(f"   Kernel Time: {kernel_time:.2f} ms / 8 ms (target)")
        print(f"   Memory Transfer: {total_transfer:.2f} ms / 3 ms (target)")
        if 'throughput' in locals():
            print(f"   Memory Throughput: {throughput:.2f}% / 80% (target)")
        if 'occupancy' in locals():
            print(f"   Occupancy: {occupancy:.2f}% / 70% (target)")

if __name__ == "__main__":
    measure_performance()

