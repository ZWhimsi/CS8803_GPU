#!/usr/bin/env python3

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

def analyze_final_grading():
    print("=" * 80)
    print("ASSIGNMENT 4 FINAL GRADE CALCULATION")
    print("Based on README: Task 1 uses NUM_CYCLES, Task 2 uses NUM_STALL_CYCLES")
    print("Grading: 50% base + 40% tolerance + 10% exact match per benchmark")
    print("=" * 80)
    
    # Task 1: Uses NUM_CYCLES values (6 points total)
    print("\nTASK 1 (CC) - COMPUTE CORE - NUM_CYCLES:")
    print("-" * 60)
    print("Benchmark        | Your Value | Ref Value | Error % | Status")
    print("-" * 60)
    
    task1_benchmarks = ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]
    task1_points = 0.0
    
    for benchmark in task1_benchmarks:
        your_val = your_results["CC"][benchmark]["NUM_CYCLES"]
        ref_val = ref_results["CC"][benchmark]["NUM_CYCLES"]
        error = calculate_percentage_error(your_val, ref_val)
        
        # Each benchmark is worth 6/8 = 0.75 points
        benchmark_points = 0.75
        
        if check_exact_match(your_val, ref_val):
            status = "EXACT MATCH"
            # 50% base + 40% tolerance + 10% exact = 100% = 0.75 points
            points = benchmark_points * 1.0
        elif check_tolerance(your_val, ref_val):
            status = "WITHIN TOLERANCE"
            # 50% base + 40% tolerance = 90% = 0.675 points
            points = benchmark_points * 0.9
        else:
            status = "OUTSIDE TOLERANCE"
            # 50% base only = 0.375 points
            points = benchmark_points * 0.5
        
        task1_points += points
        print(f"{benchmark:15} | {your_val:10.0f} | {ref_val:9.0f} | {error:6.2f}% | {status} ({points:.3f} pts)")
    
    # Task 2: Uses NUM_STALL_CYCLES values (6 points total)
    print("\nTASK 2 (TC) - TENSOR CORE - NUM_STALL_CYCLES:")
    print("-" * 60)
    print("Benchmark        | Your Value | Ref Value | Error % | Status")
    print("-" * 60)
    
    task2_benchmarks = ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]
    task2_points = 0.0
    
    for benchmark in task2_benchmarks:
        your_val = your_results["TC"][benchmark]["NUM_STALL_CYCLES"]
        ref_val = ref_results["TC"][benchmark]["NUM_STALL_CYCLES"]
        error = calculate_percentage_error(your_val, ref_val)
        
        # Each benchmark is worth 6/8 = 0.75 points
        benchmark_points = 0.75
        
        if check_exact_match(your_val, ref_val):
            status = "EXACT MATCH"
            # 50% base + 40% tolerance + 10% exact = 100% = 0.75 points
            points = benchmark_points * 1.0
        elif check_tolerance(your_val, ref_val):
            status = "WITHIN TOLERANCE"
            # 50% base + 40% tolerance = 90% = 0.675 points
            points = benchmark_points * 0.9
        else:
            status = "OUTSIDE TOLERANCE"
            # 50% base only = 0.375 points
            points = benchmark_points * 0.5
        
        task2_points += points
        print(f"{benchmark:15} | {your_val:10.0f} | {ref_val:9.0f} | {error:6.2f}% | {status} ({points:.3f} pts)")
    
    # Report points (3 points - pass/fail)
    report_points = 3.0  # Assuming report is submitted
    
    # Total points
    total_points = task1_points + task2_points + report_points
    max_points = 15.0
    percentage = (total_points / max_points) * 100
    
    print("\n" + "=" * 80)
    print("FINAL GRADE CALCULATION:")
    print("-" * 60)
    print(f"Task 1 (Compute Core): {task1_points:.2f}/6.0 points")
    print(f"Task 2 (Tensor Core):  {task2_points:.2f}/6.0 points") 
    print(f"Report:                {report_points:.1f}/3.0 points")
    print(f"TOTAL:                 {total_points:.2f}/15.0 points")
    print(f"PERCENTAGE:            {percentage:.1f}%")
    
    if percentage >= 100:
        print("\n🎉 PERFECT SCORE! 100%")
    elif percentage >= 95:
        print("\n🎉 EXCELLENT! A+ (95%+)")
    elif percentage >= 90:
        print("\n🎉 GREAT! A (90%+)")
    elif percentage >= 85:
        print("\n👍 GOOD! B+ (85%+)")
    elif percentage >= 80:
        print("\n👍 GOOD! B (80%+)")
    else:
        print("\n❌ NEEDS IMPROVEMENT")
    
    # Check if we can get 100%
    print("\n" + "=" * 80)
    print("100% ACHIEVEMENT ANALYSIS:")
    print("-" * 60)
    
    if task1_points >= 5.9 and task2_points >= 5.9:
        print("✅ YES, YOU CAN GET 100%!")
        print("   - Task 1: Nearly perfect performance")
        print("   - Task 2: Nearly perfect performance") 
        print("   - Report: Full points (3/3)")
        print("\n🎯 RECOMMENDATION: Submit as-is for 100%!")
    else:
        print("⚠️  ANALYSIS:")
        if task1_points >= 5.9:
            print(f"   ✅ Task 1: Excellent ({task1_points:.2f}/6.0)")
        else:
            print(f"   ⚠️  Task 1: Needs improvement ({task1_points:.2f}/6.0)")
            
        if task2_points >= 5.9:
            print(f"   ✅ Task 2: Excellent ({task2_points:.2f}/6.0)")
        else:
            print(f"   ⚠️  Task 2: Needs improvement ({task2_points:.2f}/6.0)")
    
    print(f"\n📊 SUMMARY:")
    print(f"   - Your current grade: {percentage:.1f}%")
    print(f"   - Task 1: {task1_points:.2f}/6.0 ({task1_points/6*100:.1f}%)")
    print(f"   - Task 2: {task2_points:.2f}/6.0 ({task2_points/6*100:.1f}%)")
    print(f"   - Report: 3.0/3.0 (100%)")

if __name__ == "__main__":
    analyze_final_grading()
