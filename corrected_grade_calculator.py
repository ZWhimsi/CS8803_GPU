#!/usr/bin/env python3

import json

# Your results
your_results = {
    "CC": {
        "gemm_float": {"NUM_CYCLES": 109220.0},
        "gemm_half": {"NUM_CYCLES": 68512.0},
        "cnn_float": {"NUM_CYCLES": 2536533.0},
        "cnn_half": {"NUM_CYCLES": 3380490.0},
        "ffn_float": {"NUM_CYCLES": 8256909.0},
        "ffn_half": {"NUM_CYCLES": 1823943.0},
        "gpt2_float": {"NUM_CYCLES": 10002387.0},
        "gpt2_half": {"NUM_CYCLES": 5743191.0}
    },
    "TC": {
        "gemm_float": {"NUM_STALL_CYCLES": 362254.0},
        "gemm_half": {"NUM_STALL_CYCLES": 281982.0},
        "cnn_float": {"NUM_STALL_CYCLES": 7049370.0},
        "cnn_half": {"NUM_STALL_CYCLES": 13665083.0},
        "ffn_float": {"NUM_STALL_CYCLES": 42174435.0},
        "ffn_half": {"NUM_STALL_CYCLES": 9680713.0},
        "gpt2_float": {"NUM_STALL_CYCLES": 34780934.0},
        "gpt2_half": {"NUM_STALL_CYCLES": 24877254.0}
    }
}

# Reference results
ref_results = {
    "CC": {
        "gemm_float": {"NUM_CYCLES": 109220.0},
        "gemm_half": {"NUM_CYCLES": 68512.0},
        "cnn_float": {"NUM_CYCLES": 2536533.0},
        "cnn_half": {"NUM_CYCLES": 3380490.0},
        "ffn_float": {"NUM_CYCLES": 8256909.0},
        "ffn_half": {"NUM_CYCLES": 1823943.0},
        "gpt2_float": {"NUM_CYCLES": 10002387.0},
        "gpt2_half": {"NUM_CYCLES": 5743191.0}
    },
    "TC": {
        "gemm_float": {"NUM_STALL_CYCLES": 362254.0},
        "gemm_half": {"NUM_STALL_CYCLES": 283688.0},
        "cnn_float": {"NUM_STALL_CYCLES": 7049370.0},
        "cnn_half": {"NUM_STALL_CYCLES": 14074545.0},
        "ffn_float": {"NUM_STALL_CYCLES": 42174435.0},
        "ffn_half": {"NUM_STALL_CYCLES": 10188086.0},
        "gpt2_float": {"NUM_STALL_CYCLES": 34780934.0},
        "gpt2_half": {"NUM_STALL_CYCLES": 26266192.0}
    }
}

def calculate_percentage_error(your_val, ref_val):
    """Calculate percentage error between your value and reference value"""
    if ref_val == 0:
        return 0 if your_val == 0 else float('inf')
    return abs((your_val - ref_val) / ref_val) * 100

def check_tolerance(your_val, ref_val, tolerance=5.0):
    """Check if your value is within tolerance of reference value"""
    error = calculate_percentage_error(your_val, ref_val)
    return error <= tolerance

def check_exact_match(your_val, ref_val):
    """Check if your value exactly matches reference value"""
    return your_val == ref_val

def analyze_grading():
    print("=" * 80)
    print("ASSIGNMENT 4 GRADING CALCULATION (CORRECTED)")
    print("Based on README: Task 1 uses NUM_CYCLES, Task 2 uses NUM_STALL_CYCLES")
    print("=" * 80)
    
    # Task 1: Uses NUM_CYCLES values
    print("\nTASK 1 (CC) - COMPUTE CORE - NUM_CYCLES:")
    print("-" * 60)
    print("Benchmark        | Your Value | Ref Value | Error % | Status")
    print("-" * 60)
    
    task1_exact_matches = 0
    task1_tolerance_matches = 0
    task1_total = 8
    
    for benchmark in ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]:
        your_val = your_results["CC"][benchmark]["NUM_CYCLES"]
        ref_val = ref_results["CC"][benchmark]["NUM_CYCLES"]
        error = calculate_percentage_error(your_val, ref_val)
        
        if check_exact_match(your_val, ref_val):
            status = "EXACT MATCH"
            task1_exact_matches += 1
            task1_tolerance_matches += 1
        elif check_tolerance(your_val, ref_val):
            status = "WITHIN TOLERANCE"
            task1_tolerance_matches += 1
        else:
            status = "OUTSIDE TOLERANCE"
        
        print(f"{benchmark:15} | {your_val:10.0f} | {ref_val:9.0f} | {error:6.2f}% | {status}")
    
    # Task 2: Uses NUM_STALL_CYCLES values
    print("\nTASK 2 (TC) - TENSOR CORE - NUM_STALL_CYCLES:")
    print("-" * 60)
    print("Benchmark        | Your Value | Ref Value | Error % | Status")
    print("-" * 60)
    
    task2_exact_matches = 0
    task2_tolerance_matches = 0
    task2_total = 8
    
    for benchmark in ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]:
        your_val = your_results["TC"][benchmark]["NUM_STALL_CYCLES"]
        ref_val = ref_results["TC"][benchmark]["NUM_STALL_CYCLES"]
        error = calculate_percentage_error(your_val, ref_val)
        
        if check_exact_match(your_val, ref_val):
            status = "EXACT MATCH"
            task2_exact_matches += 1
            task2_tolerance_matches += 1
        elif check_tolerance(your_val, ref_val):
            status = "WITHIN TOLERANCE"
            task2_tolerance_matches += 1
        else:
            status = "OUTSIDE TOLERANCE"
        
        print(f"{benchmark:15} | {your_val:10.0f} | {ref_val:9.0f} | {error:6.2f}% | {status}")
    
    # Calculate points based on README rubric
    print("\n" + "=" * 80)
    print("GRADING CALCULATION:")
    print("-" * 60)
    
    # Task 1 Points (6 points total)
    task1_exact_points = task1_exact_matches * (6.0 / task1_total) * 0.1  # 10% for exact match
    task1_tolerance_points = task1_tolerance_matches * (6.0 / task1_total) * 0.4  # 40% for within tolerance
    task1_total_points = min(6.0, task1_exact_points + task1_tolerance_points)
    
    print(f"Task 1 (CC) - NUM_CYCLES:")
    print(f"  Exact matches: {task1_exact_matches}/{task1_total} ({task1_exact_points:.2f} points)")
    print(f"  Within tolerance: {task1_tolerance_matches}/{task1_total} ({task1_tolerance_points:.2f} points)")
    print(f"  Total Task 1 points: {task1_total_points:.2f}/6.0")
    
    # Task 2 Points (6 points total)
    task2_exact_points = task2_exact_matches * (6.0 / task2_total) * 0.1  # 10% for exact match
    task2_tolerance_points = task2_tolerance_matches * (6.0 / task2_total) * 0.4  # 40% for within tolerance
    task2_total_points = min(6.0, task2_exact_points + task2_tolerance_points)
    
    print(f"\nTask 2 (TC) - NUM_STALL_CYCLES:")
    print(f"  Exact matches: {task2_exact_matches}/{task2_total} ({task2_exact_points:.2f} points)")
    print(f"  Within tolerance: {task2_tolerance_matches}/{task2_total} ({task2_tolerance_points:.2f} points)")
    print(f"  Total Task 2 points: {task2_total_points:.2f}/6.0")
    
    # Report points (3 points - pass/fail)
    report_points = 3.0  # Assuming report is submitted
    
    # Total points
    total_points = task1_total_points + task2_total_points + report_points
    max_points = 15.0
    percentage = (total_points / max_points) * 100
    
    print(f"\nReport: {report_points:.1f}/3.0 (assumed submitted)")
    print(f"\nTOTAL GRADE: {total_points:.2f}/{max_points} = {percentage:.1f}%")
    
    if percentage >= 100:
        print("🎉 PERFECT SCORE! 100%")
    elif percentage >= 95:
        print("🎉 EXCELLENT! A+")
    elif percentage >= 90:
        print("🎉 GREAT! A")
    elif percentage >= 85:
        print("👍 GOOD! B+")
    else:
        print("❌ NEEDS IMPROVEMENT")
    
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN:")
    print("-" * 60)
    print(f"Task 1 (Compute Core): {task1_total_points:.2f}/6.0 points")
    print(f"Task 2 (Tensor Core):  {task2_total_points:.2f}/6.0 points") 
    print(f"Report:                {report_points:.1f}/3.0 points")
    print(f"TOTAL:                 {total_points:.2f}/15.0 points")
    
    # Check if we can get 100%
    if task1_total_points >= 5.9 and task2_total_points >= 5.9:
        print("\n✅ YES, YOU CAN GET 100%!")
        print("   - Task 1: Nearly perfect (within tolerance on all benchmarks)")
        print("   - Task 2: Nearly perfect (within tolerance on all benchmarks)")
        print("   - Report: Full points (3/3)")
    else:
        print("\n⚠️  MINOR IMPROVEMENTS NEEDED:")
        if task1_total_points < 5.9:
            print(f"   - Task 1 needs improvement: {task1_total_points:.2f}/6.0")
        if task2_total_points < 5.9:
            print(f"   - Task 2 needs improvement: {task2_total_points:.2f}/6.0")

if __name__ == "__main__":
    analyze_grading()
