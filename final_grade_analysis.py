#!/usr/bin/env python3

# Current results from the user
current_results = {
    "CC": {
        "gemm_float": {"NUM_CYCLES": 109220.0, "NUM_STALL_CYCLES": 540430.0},
        "gemm_half": {"NUM_CYCLES": 68512.0, "NUM_STALL_CYCLES": 316464.0},
        "cnn_float": {"NUM_CYCLES": 2536533.0, "NUM_STALL_CYCLES": 13602394.0},
        "cnn_half": {"NUM_CYCLES": 3380490.0, "NUM_STALL_CYCLES": 17524365.0},
        "ffn_float": {"NUM_CYCLES": 8256909.0, "NUM_STALL_CYCLES": 52112355.0},
        "ffn_half": {"NUM_CYCLES": 1823943.0, "NUM_STALL_CYCLES": 10209238.0},
        "gpt2_float": {"NUM_CYCLES": 10002387.0, "NUM_STALL_CYCLES": 39700934.0},
        "gpt2_half": {"NUM_CYCLES": 5743191.0, "NUM_STALL_CYCLES": 23369854.0}
    },
    "TC": {
        "gemm_float": {"NUM_CYCLES": 109220.0, "NUM_STALL_CYCLES": 540430.0},
        "gemm_half": {"NUM_CYCLES": 69100.0, "NUM_STALL_CYCLES": 320766.0},
        "cnn_float": {"NUM_CYCLES": 2536533.0, "NUM_STALL_CYCLES": 13602394.0},
        "cnn_half": {"NUM_CYCLES": 3613251.0, "NUM_STALL_CYCLES": 19379618.0},
        "ffn_float": {"NUM_CYCLES": 8256909.0, "NUM_STALL_CYCLES": 52112355.0},
        "ffn_half": {"NUM_CYCLES": 1929424.0, "NUM_STALL_CYCLES": 11043145.0},
        "gpt2_float": {"NUM_CYCLES": 10002387.0, "NUM_STALL_CYCLES": 39700934.0},
        "gpt2_half": {"NUM_CYCLES": 6768134.0, "NUM_STALL_CYCLES": 31426058.0}
    }
}

# Reference results
reference_results = {
    "CC": {
        "gemm_float": {"NUM_CYCLES": 109220.0, "NUM_STALL_CYCLES": 540430.0},
        "gemm_half": {"NUM_CYCLES": 68512.0, "NUM_STALL_CYCLES": 316464.0},
        "cnn_float": {"NUM_CYCLES": 2536533.0, "NUM_STALL_CYCLES": 13602394.0},
        "cnn_half": {"NUM_CYCLES": 3380490.0, "NUM_STALL_CYCLES": 17524365.0},
        "ffn_float": {"NUM_CYCLES": 8256909.0, "NUM_STALL_CYCLES": 52112355.0},
        "ffn_half": {"NUM_CYCLES": 1823943.0, "NUM_STALL_CYCLES": 10209238.0},
        "gpt2_float": {"NUM_CYCLES": 10002387.0, "NUM_STALL_CYCLES": 39700934.0},
        "gpt2_half": {"NUM_CYCLES": 5743191.0, "NUM_STALL_CYCLES": 23369854.0}
    },
    "TC": {
        "gemm_float": {"NUM_CYCLES": 109220.0, "NUM_STALL_CYCLES": 540430.0},
        "gemm_half": {"NUM_CYCLES": 69100.0, "NUM_STALL_CYCLES": 283688.0},
        "cnn_float": {"NUM_CYCLES": 2536533.0, "NUM_STALL_CYCLES": 13602394.0},
        "cnn_half": {"NUM_CYCLES": 3613251.0, "NUM_STALL_CYCLES": 14074545.0},
        "ffn_float": {"NUM_CYCLES": 8256909.0, "NUM_STALL_CYCLES": 52112355.0},
        "ffn_half": {"NUM_CYCLES": 1929424.0, "NUM_STALL_CYCLES": 10188086.0},
        "gpt2_float": {"NUM_CYCLES": 10002387.0, "NUM_STALL_CYCLES": 39700934.0},
        "gpt2_half": {"NUM_CYCLES": 6768134.0, "NUM_STALL_CYCLES": 26266192.0}
    }
}

def calculate_grade():
    total_score = 0
    total_possible = 0
    
    print("=== FINAL GRADE ANALYSIS ===\n")
    
    # Task 1: CC benchmarks (NUM_CYCLES)
    print("TASK 1: Compute Core (CC) - NUM_CYCLES")
    print("-" * 50)
    task1_score = 0
    task1_possible = 0
    
    for benchmark in ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]:
        current = current_results["CC"][benchmark]["NUM_CYCLES"]
        reference = reference_results["CC"][benchmark]["NUM_CYCLES"]
        
        if current == reference:
            score = 100  # Exact match bonus
            status = "EXACT MATCH ✅"
        else:
            diff_percent = abs(current - reference) / reference * 100
            if diff_percent <= 5:
                score = 90
                status = f"Within ±5% ({diff_percent:.2f}%) ✅"
            else:
                score = 0
                status = f"Outside ±5% ({diff_percent:.2f}%) ❌"
        
        task1_score += score
        task1_possible += 100
        print(f"{benchmark:12}: {current:>10} vs {reference:>10} - {status}")
    
    print(f"\nTask 1 Score: {task1_score}/{task1_possible} = {task1_score/task1_possible*100:.1f}%")
    total_score += task1_score
    total_possible += task1_possible
    
    # Task 2: TC benchmarks (NUM_STALL_CYCLES)
    print("\nTASK 2: Tensor Core (TC) - NUM_STALL_CYCLES")
    print("-" * 50)
    task2_score = 0
    task2_possible = 0
    
    for benchmark in ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]:
        current = current_results["TC"][benchmark]["NUM_STALL_CYCLES"]
        reference = reference_results["TC"][benchmark]["NUM_STALL_CYCLES"]
        
        if current == reference:
            score = 100  # Exact match bonus
            status = "EXACT MATCH ✅"
        else:
            diff_percent = abs(current - reference) / reference * 100
            if diff_percent <= 5:
                score = 90
                status = f"Within ±5% ({diff_percent:.2f}%) ✅"
            else:
                score = 0
                status = f"Outside ±5% ({diff_percent:.2f}%) ❌"
        
        task2_score += score
        task2_possible += 100
        print(f"{benchmark:12}: {current:>10} vs {reference:>10} - {status}")
    
    print(f"\nTask 2 Score: {task2_score}/{task2_possible} = {task2_score/task2_possible*100:.1f}%")
    total_score += task2_score
    total_possible += task2_possible
    
    # Overall grade
    overall_grade = total_score / total_possible * 100
    print(f"\n{'='*60}")
    print(f"OVERALL GRADE: {total_score}/{total_possible} = {overall_grade:.1f}%")
    
    if overall_grade >= 100:
        print("🎉 PERFECT SCORE! 100% A+")
    elif overall_grade >= 90:
        print("🎯 EXCELLENT! A+ Grade")
    elif overall_grade >= 80:
        print("👍 GOOD! A Grade")
    else:
        print("📚 Needs improvement")
    
    return overall_grade

if __name__ == "__main__":
    calculate_grade()
