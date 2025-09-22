#!/usr/bin/env python3
"""
Performance measurement script for CUDA bitonic sort optimization
Measures memory throughput and occupancy using NSight Compute profiler
"""

import subprocess
import sys
import os

def run_ncu_metric(metric, size):
    """Run ncu command to measure a specific metric"""
    cmd = f"ncu --metric {metric} --print-summary per-gpu ./a.out {size}"
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Error running ncu: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("ncu command timed out")
        return None

def extract_metric_value(output, metric_name):
    """Extract metric value from ncu output"""
    lines = output.split('\n')
    for line in lines:
        if metric_name in line and '%' in line:
            # Extract percentage value
            parts = line.split()
            for part in parts:
                if '%' in part:
                    return float(part.replace('%', ''))
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python measure_performance.py <array_size>")
        sys.exit(1)
    
    size = sys.argv[1]
    print(f"Measuring performance for array size: {size}")
    print("=" * 60)
    
    # Measure Memory Throughput
    print("\n1. Measuring Memory Throughput...")
    memory_output = run_ncu_metric("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", size)
    if memory_output:
        memory_throughput = extract_metric_value(memory_output, "gpu__compute_memory_throughput")
        if memory_throughput:
            print(f"Memory Throughput: {memory_throughput:.2f}%")
            if memory_throughput >= 80:
                print("✅ PASSED (≥80%)")
            else:
                print("❌ FAILED (<80%)")
        else:
            print("Could not extract memory throughput value")
    else:
        print("Failed to measure memory throughput")
    
    # Measure Achieved Occupancy
    print("\n2. Measuring Achieved Occupancy...")
    occupancy_output = run_ncu_metric("sm__warps_active.avg.pct_of_peak_sustained_active", size)
    if occupancy_output:
        occupancy = extract_metric_value(occupancy_output, "sm__warps_active")
        if occupancy:
            print(f"Achieved Occupancy: {occupancy:.2f}%")
            if occupancy >= 70:
                print("✅ PASSED (≥70%)")
            else:
                print("❌ FAILED (<70%)")
        else:
            print("Could not extract occupancy value")
    else:
        print("Failed to measure occupancy")
    
    print("\n" + "=" * 60)
    print("Performance measurement complete!")

if __name__ == "__main__":
    main()
