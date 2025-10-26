#!/usr/bin/env python3

import json

# Your results
your_results = {
    "CC": {
        "gemm_float": {
            "NUM_CYCLES": 109220.0,
            "NUM_INSTRS_RETIRED": 485376.0,
            "NUM_STALL_CYCLES": 362254.0,
            "NUM_MEM_REQUESTS": 9503.0,
            "NUM_MEM_RESPONSES": 9503.0,
            "AVG_RESPONSE_LATENCY": 185.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.444021,
            "CACHE_NUM_ACCESSES": 54234.0,
            "CACHE_NUM_HITS": 10418.0,
            "CACHE_HIT_RATE_PERC": 19.21,
            "MISSES_PER_1000_INSTR": 90.27
        },
        "gemm_half": {
            "NUM_CYCLES": 68512.0,
            "NUM_INSTRS_RETIRED": 254208.0,
            "NUM_STALL_CYCLES": 277680.0,
            "NUM_MEM_REQUESTS": 6445.0,
            "NUM_MEM_RESPONSES": 6445.0,
            "AVG_RESPONSE_LATENCY": 179.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 3.710416,
            "CACHE_NUM_ACCESSES": 54824.0,
            "CACHE_NUM_HITS": 7272.0,
            "CACHE_HIT_RATE_PERC": 13.26,
            "MISSES_PER_1000_INSTR": 187.06
        },
        "cnn_float": {
            "NUM_CYCLES": 2536533.0,
            "NUM_INSTRS_RETIRED": 12896154.0,
            "NUM_STALL_CYCLES": 7049370.0,
            "NUM_MEM_REQUESTS": 240247.0,
            "NUM_MEM_RESPONSES": 240247.0,
            "AVG_RESPONSE_LATENCY": 140.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 5.084166,
            "CACHE_NUM_ACCESSES": 735455.0,
            "CACHE_NUM_HITS": 246953.0,
            "CACHE_HIT_RATE_PERC": 33.58,
            "MISSES_PER_1000_INSTR": 37.88
        },
        "cnn_half": {
            "NUM_CYCLES": 3380490.0,
            "NUM_INSTRS_RETIRED": 14857416.0,
            "NUM_STALL_CYCLES": 11809830.0,
            "NUM_MEM_REQUESTS": 437075.0,
            "NUM_MEM_RESPONSES": 437075.0,
            "AVG_RESPONSE_LATENCY": 119.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.395048,
            "CACHE_NUM_ACCESSES": 1966978.0,
            "CACHE_NUM_HITS": 313582.0,
            "CACHE_HIT_RATE_PERC": 15.94,
            "MISSES_PER_1000_INSTR": 111.28
        },
        "ffn_float": {
            "NUM_CYCLES": 8256909.0,
            "NUM_INSTRS_RETIRED": 22750114.0,
            "NUM_STALL_CYCLES": 42174435.0,
            "NUM_MEM_REQUESTS": 1122149.0,
            "NUM_MEM_RESPONSES": 1122137.0,
            "AVG_RESPONSE_LATENCY": 197.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 2.755282,
            "CACHE_NUM_ACCESSES": 6842459.0,
            "CACHE_NUM_HITS": 3887763.0,
            "CACHE_HIT_RATE_PERC": 56.82,
            "MISSES_PER_1000_INSTR": 129.88
        },
        "ffn_half": {
            "NUM_CYCLES": 1823943.0,
            "NUM_INSTRS_RETIRED": 5508992.0,
            "NUM_STALL_CYCLES": 8846806.0,
            "NUM_MEM_REQUESTS": 237001.0,
            "NUM_MEM_RESPONSES": 236999.0,
            "AVG_RESPONSE_LATENCY": 180.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 3.020375,
            "CACHE_NUM_ACCESSES": 928974.0,
            "CACHE_NUM_HITS": 195341.0,
            "CACHE_HIT_RATE_PERC": 21.03,
            "MISSES_PER_1000_INSTR": 133.17
        },
        "gpt2_float": {
            "NUM_CYCLES": 10002387.0,
            "NUM_INSTRS_RETIRED": 43232239.0,
            "NUM_STALL_CYCLES": 34780934.0,
            "NUM_MEM_REQUESTS": 819158.0,
            "NUM_MEM_RESPONSES": 819158.0,
            "AVG_RESPONSE_LATENCY": 202.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.322192,
            "CACHE_NUM_ACCESSES": 33211829.0,
            "CACHE_NUM_HITS": 3472771.0,
            "CACHE_HIT_RATE_PERC": 10.46,
            "MISSES_PER_1000_INSTR": 687.89
        },
        "gpt2_half": {
            "NUM_CYCLES": 5743191.0,
            "NUM_INSTRS_RETIRED": 27506706.0,
            "NUM_STALL_CYCLES": 16821050.0,
            "NUM_MEM_REQUESTS": 380889.0,
            "NUM_MEM_RESPONSES": 380889.0,
            "AVG_RESPONSE_LATENCY": 201.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.789446,
            "CACHE_NUM_ACCESSES": 15494615.0,
            "CACHE_NUM_HITS": 629148.0,
            "CACHE_HIT_RATE_PERC": 4.06,
            "MISSES_PER_1000_INSTR": 540.43
        }
    },
    "TC": {
        "gemm_float": {
            "NUM_CYCLES": 109220.0,
            "NUM_INSTRS_RETIRED": 485376.0,
            "NUM_STALL_CYCLES": 362254.0,
            "NUM_MEM_REQUESTS": 9503.0,
            "NUM_MEM_RESPONSES": 9503.0,
            "AVG_RESPONSE_LATENCY": 185.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.444021,
            "CACHE_NUM_ACCESSES": 54234.0,
            "CACHE_NUM_HITS": 10418.0,
            "CACHE_HIT_RATE_PERC": 19.21,
            "MISSES_PER_1000_INSTR": 90.27
        },
        "gemm_half": {
            "NUM_CYCLES": 69100.0,
            "NUM_INSTRS_RETIRED": 254208.0,
            "NUM_STALL_CYCLES": 281982.0,
            "NUM_MEM_REQUESTS": 6445.0,
            "NUM_MEM_RESPONSES": 6445.0,
            "AVG_RESPONSE_LATENCY": 179.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 3.678842,
            "CACHE_NUM_ACCESSES": 54824.0,
            "CACHE_NUM_HITS": 7302.0,
            "CACHE_HIT_RATE_PERC": 13.32,
            "MISSES_PER_1000_INSTR": 186.94
        },
        "cnn_float": {
            "NUM_CYCLES": 2536533.0,
            "NUM_INSTRS_RETIRED": 12896154.0,
            "NUM_STALL_CYCLES": 7049370.0,
            "NUM_MEM_REQUESTS": 240247.0,
            "NUM_MEM_RESPONSES": 240247.0,
            "AVG_RESPONSE_LATENCY": 140.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 5.084166,
            "CACHE_NUM_ACCESSES": 735455.0,
            "CACHE_NUM_HITS": 246953.0,
            "CACHE_HIT_RATE_PERC": 33.58,
            "MISSES_PER_1000_INSTR": 37.88
        },
        "cnn_half": {
            "NUM_CYCLES": 3613251.0,
            "NUM_INSTRS_RETIRED": 14857416.0,
            "NUM_STALL_CYCLES": 13665083.0,
            "NUM_MEM_REQUESTS": 431575.0,
            "NUM_MEM_RESPONSES": 431575.0,
            "AVG_RESPONSE_LATENCY": 118.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.111925,
            "CACHE_NUM_ACCESSES": 1961934.0,
            "CACHE_NUM_HITS": 309976.0,
            "CACHE_HIT_RATE_PERC": 15.8,
            "MISSES_PER_1000_INSTR": 111.19
        },
        "ffn_float": {
            "NUM_CYCLES": 8256909.0,
            "NUM_INSTRS_RETIRED": 22750114.0,
            "NUM_STALL_CYCLES": 42174435.0,
            "NUM_MEM_REQUESTS": 1122149.0,
            "NUM_MEM_RESPONSES": 1122137.0,
            "AVG_RESPONSE_LATENCY": 197.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 2.755282,
            "CACHE_NUM_ACCESSES": 6842459.0,
            "CACHE_NUM_HITS": 3887763.0,
            "CACHE_HIT_RATE_PERC": 56.82,
            "MISSES_PER_1000_INSTR": 129.88
        },
        "ffn_half": {
            "NUM_CYCLES": 1929424.0,
            "NUM_INSTRS_RETIRED": 5508992.0,
            "NUM_STALL_CYCLES": 9680713.0,
            "NUM_MEM_REQUESTS": 235492.0,
            "NUM_MEM_RESPONSES": 235488.0,
            "AVG_RESPONSE_LATENCY": 180.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 2.855252,
            "CACHE_NUM_ACCESSES": 927900.0,
            "CACHE_NUM_HITS": 194730.0,
            "CACHE_HIT_RATE_PERC": 20.99,
            "MISSES_PER_1000_INSTR": 133.09
        },
        "gpt2_float": {
            "NUM_CYCLES": 10002387.0,
            "NUM_INSTRS_RETIRED": 43232239.0,
            "NUM_STALL_CYCLES": 34780934.0,
            "NUM_MEM_REQUESTS": 819158.0,
            "NUM_MEM_RESPONSES": 819158.0,
            "AVG_RESPONSE_LATENCY": 202.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.322192,
            "CACHE_NUM_ACCESSES": 33211829.0,
            "CACHE_NUM_HITS": 3472771.0,
            "CACHE_HIT_RATE_PERC": 10.46,
            "MISSES_PER_1000_INSTR": 687.89
        },
        "gpt2_half": {
            "NUM_CYCLES": 6768134.0,
            "NUM_INSTRS_RETIRED": 27506706.0,
            "NUM_STALL_CYCLES": 24877254.0,
            "NUM_MEM_REQUESTS": 383045.0,
            "NUM_MEM_RESPONSES": 383045.0,
            "AVG_RESPONSE_LATENCY": 202.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.064149,
            "CACHE_NUM_ACCESSES": 15496812.0,
            "CACHE_NUM_HITS": 653132.0,
            "CACHE_HIT_RATE_PERC": 4.21,
            "MISSES_PER_1000_INSTR": 539.64
        }
    }
}

# Reference results
ref_results = {
    "CC": {
        "gemm_float": {
            "NUM_CYCLES": 109220.0,
            "NUM_INSTRS_RETIRED": 485376.0,
            "NUM_STALL_CYCLES": 362254.0,
            "NUM_MEM_REQUESTS": 9503.0,
            "NUM_MEM_RESPONSES": 9503.0,
            "AVG_RESPONSE_LATENCY": 185.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.444021,
            "CACHE_NUM_ACCESSES": 54234.0,
            "CACHE_NUM_HITS": 10418.0,
            "CACHE_HIT_RATE_PERC": 19.21,
            "MISSES_PER_1000_INSTR": 90.27
        },
        "gemm_half": {
            "NUM_CYCLES": 68512.0,
            "NUM_INSTRS_RETIRED": 254208.0,
            "NUM_STALL_CYCLES": 277680.0,
            "NUM_MEM_REQUESTS": 6445.0,
            "NUM_MEM_RESPONSES": 6445.0,
            "AVG_RESPONSE_LATENCY": 179.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 3.710416,
            "CACHE_NUM_ACCESSES": 54824.0,
            "CACHE_NUM_HITS": 7272.0,
            "CACHE_HIT_RATE_PERC": 13.26,
            "MISSES_PER_1000_INSTR": 187.06
        },
        "cnn_float": {
            "NUM_CYCLES": 2536533.0,
            "NUM_INSTRS_RETIRED": 12896154.0,
            "NUM_STALL_CYCLES": 7049370.0,
            "NUM_MEM_REQUESTS": 240247.0,
            "NUM_MEM_RESPONSES": 240247.0,
            "AVG_RESPONSE_LATENCY": 140.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 5.084166,
            "CACHE_NUM_ACCESSES": 735455.0,
            "CACHE_NUM_HITS": 246953.0,
            "CACHE_HIT_RATE_PERC": 33.58,
            "MISSES_PER_1000_INSTR": 37.88
        },
        "cnn_half": {
            "NUM_CYCLES": 3380490.0,
            "NUM_INSTRS_RETIRED": 14857416.0,
            "NUM_STALL_CYCLES": 11809830.0,
            "NUM_MEM_REQUESTS": 437075.0,
            "NUM_MEM_RESPONSES": 437075.0,
            "AVG_RESPONSE_LATENCY": 119.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.395048,
            "CACHE_NUM_ACCESSES": 1966978.0,
            "CACHE_NUM_HITS": 313582.0,
            "CACHE_HIT_RATE_PERC": 15.94,
            "MISSES_PER_1000_INSTR": 111.28
        },
        "ffn_float": {
            "NUM_CYCLES": 8256909.0,
            "NUM_INSTRS_RETIRED": 22750114.0,
            "NUM_STALL_CYCLES": 42174435.0,
            "NUM_MEM_REQUESTS": 1122149.0,
            "NUM_MEM_RESPONSES": 1122137.0,
            "AVG_RESPONSE_LATENCY": 197.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 2.755282,
            "CACHE_NUM_ACCESSES": 6842459.0,
            "CACHE_NUM_HITS": 3887763.0,
            "CACHE_HIT_RATE_PERC": 56.82,
            "MISSES_PER_1000_INSTR": 129.88
        },
        "ffn_half": {
            "NUM_CYCLES": 1823943.0,
            "NUM_INSTRS_RETIRED": 5508992.0,
            "NUM_STALL_CYCLES": 8846806.0,
            "NUM_MEM_REQUESTS": 237001.0,
            "NUM_MEM_RESPONSES": 236999.0,
            "AVG_RESPONSE_LATENCY": 180.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 3.020375,
            "CACHE_NUM_ACCESSES": 928974.0,
            "CACHE_NUM_HITS": 195341.0,
            "CACHE_HIT_RATE_PERC": 21.03,
            "MISSES_PER_1000_INSTR": 133.17
        },
        "gpt2_float": {
            "NUM_CYCLES": 10002387.0,
            "NUM_INSTRS_RETIRED": 43232239.0,
            "NUM_STALL_CYCLES": 34780934.0,
            "NUM_MEM_REQUESTS": 819158.0,
            "NUM_MEM_RESPONSES": 819158.0,
            "AVG_RESPONSE_LATENCY": 202.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.322192,
            "CACHE_NUM_ACCESSES": 33211829.0,
            "CACHE_NUM_HITS": 3472771.0,
            "CACHE_HIT_RATE_PERC": 10.46,
            "MISSES_PER_1000_INSTR": 687.89
        },
        "gpt2_half": {
            "NUM_CYCLES": 5743191.0,
            "NUM_INSTRS_RETIRED": 27506706.0,
            "NUM_STALL_CYCLES": 16821050.0,
            "NUM_MEM_REQUESTS": 380889.0,
            "NUM_MEM_RESPONSES": 380889.0,
            "AVG_RESPONSE_LATENCY": 201.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.789446,
            "CACHE_NUM_ACCESSES": 15494615.0,
            "CACHE_NUM_HITS": 629148.0,
            "CACHE_HIT_RATE_PERC": 4.06,
            "MISSES_PER_1000_INSTR": 540.43
        }
    },
    "TC": {
        "gemm_float": {
            "NUM_CYCLES": 109220.0,
            "NUM_INSTRS_RETIRED": 485376.0,
            "NUM_STALL_CYCLES": 362254.0,
            "NUM_MEM_REQUESTS": 9503.0,
            "NUM_MEM_RESPONSES": 9503.0,
            "AVG_RESPONSE_LATENCY": 185.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.444021,
            "CACHE_NUM_ACCESSES": 54234.0,
            "CACHE_NUM_HITS": 10418.0,
            "CACHE_HIT_RATE_PERC": 19.21,
            "MISSES_PER_1000_INSTR": 90.27
        },
        "gemm_half": {
            "NUM_CYCLES": 69344.0,
            "NUM_INSTRS_RETIRED": 254208.0,
            "NUM_STALL_CYCLES": 283688.0,
            "NUM_MEM_REQUESTS": 6434.0,
            "NUM_MEM_RESPONSES": 6434.0,
            "AVG_RESPONSE_LATENCY": 179.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 3.665898,
            "CACHE_NUM_ACCESSES": 54813.0,
            "CACHE_NUM_HITS": 7256.0,
            "CACHE_HIT_RATE_PERC": 13.24,
            "MISSES_PER_1000_INSTR": 187.08
        },
        "cnn_float": {
            "NUM_CYCLES": 2536533.0,
            "NUM_INSTRS_RETIRED": 12896154.0,
            "NUM_STALL_CYCLES": 7049370.0,
            "NUM_MEM_REQUESTS": 240247.0,
            "NUM_MEM_RESPONSES": 240247.0,
            "AVG_RESPONSE_LATENCY": 140.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 5.084166,
            "CACHE_NUM_ACCESSES": 735455.0,
            "CACHE_NUM_HITS": 246953.0,
            "CACHE_HIT_RATE_PERC": 33.58,
            "MISSES_PER_1000_INSTR": 37.88
        },
        "cnn_half": {
            "NUM_CYCLES": 3677069.0,
            "NUM_INSTRS_RETIRED": 14857416.0,
            "NUM_STALL_CYCLES": 14074545.0,
            "NUM_MEM_REQUESTS": 432519.0,
            "NUM_MEM_RESPONSES": 432519.0,
            "AVG_RESPONSE_LATENCY": 119.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.040559,
            "CACHE_NUM_ACCESSES": 1962936.0,
            "CACHE_NUM_HITS": 308803.0,
            "CACHE_HIT_RATE_PERC": 15.73,
            "MISSES_PER_1000_INSTR": 111.33
        },
        "ffn_float": {
            "NUM_CYCLES": 8256909.0,
            "NUM_INSTRS_RETIRED": 22750114.0,
            "NUM_STALL_CYCLES": 42174435.0,
            "NUM_MEM_REQUESTS": 1122149.0,
            "NUM_MEM_RESPONSES": 1122137.0,
            "AVG_RESPONSE_LATENCY": 197.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 2.755282,
            "CACHE_NUM_ACCESSES": 6842459.0,
            "CACHE_NUM_HITS": 3887763.0,
            "CACHE_HIT_RATE_PERC": 56.82,
            "MISSES_PER_1000_INSTR": 129.88
        },
        "ffn_half": {
            "NUM_CYCLES": 1993474.0,
            "NUM_INSTRS_RETIRED": 5508992.0,
            "NUM_STALL_CYCLES": 10188086.0,
            "NUM_MEM_REQUESTS": 236860.0,
            "NUM_MEM_RESPONSES": 236858.0,
            "AVG_RESPONSE_LATENCY": 179.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 2.763513,
            "CACHE_NUM_ACCESSES": 928630.0,
            "CACHE_NUM_HITS": 194794.0,
            "CACHE_HIT_RATE_PERC": 20.98,
            "MISSES_PER_1000_INSTR": 133.21
        },
        "gpt2_float": {
            "NUM_CYCLES": 10002387.0,
            "NUM_INSTRS_RETIRED": 43232239.0,
            "NUM_STALL_CYCLES": 34780934.0,
            "NUM_MEM_REQUESTS": 819158.0,
            "NUM_MEM_RESPONSES": 819158.0,
            "AVG_RESPONSE_LATENCY": 202.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 4.322192,
            "CACHE_NUM_ACCESSES": 33211829.0,
            "CACHE_NUM_HITS": 3472771.0,
            "CACHE_HIT_RATE_PERC": 10.46,
            "MISSES_PER_1000_INSTR": 687.89
        },
        "gpt2_half": {
            "NUM_CYCLES": 6942994.0,
            "NUM_INSTRS_RETIRED": 27506706.0,
            "NUM_STALL_CYCLES": 26266192.0,
            "NUM_MEM_REQUESTS": 381535.0,
            "NUM_MEM_RESPONSES": 381535.0,
            "AVG_RESPONSE_LATENCY": 202.0,
            "NUM_TTIMEDOUT_REQUESTS": 0.0,
            "INSTR_PER_CYCLE": 3.961793,
            "CACHE_NUM_ACCESSES": 15495451.0,
            "CACHE_NUM_HITS": 641537.0,
            "CACHE_HIT_RATE_PERC": 4.14,
            "MISSES_PER_1000_INSTR": 540.01
        }
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

def analyze_results():
    print("=" * 80)
    print("ASSIGNMENT 4 GRADE CALCULATION")
    print("=" * 80)
    
    # Key metrics to check (based on README grading criteria)
    key_metrics = [
        "NUM_CYCLES",
        "NUM_STALL_CYCLES", 
        "INSTR_PER_CYCLE"
    ]
    
    total_tests = 0
    passed_tests = 0
    
    print("\nTASK 1 (CC) - COMPUTE CORE RESULTS:")
    print("-" * 50)
    
    for benchmark in ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]:
        print(f"\n{benchmark}:")
        your_cc = your_results["CC"][benchmark]
        ref_cc = ref_results["CC"][benchmark]
        
        benchmark_passed = True
        for metric in key_metrics:
            total_tests += 1
            your_val = your_cc[metric]
            ref_val = ref_cc[metric]
            error = calculate_percentage_error(your_val, ref_val)
            within_tol = check_tolerance(your_val, ref_val)
            
            status = "✓ PASS" if within_tol else "✗ FAIL"
            if not within_tol:
                benchmark_passed = False
            
            print(f"  {metric}: {your_val:.1f} vs {ref_val:.1f} (error: {error:.2f}%) {status}")
        
        if benchmark_passed:
            passed_tests += 3  # 3 metrics per benchmark
        else:
            passed_tests += sum(1 for metric in key_metrics if check_tolerance(your_cc[metric], ref_cc[metric]))
    
    print("\n" + "=" * 80)
    print("TASK 2 (TC) - TENSOR CORE RESULTS:")
    print("-" * 50)
    
    for benchmark in ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]:
        print(f"\n{benchmark}:")
        your_tc = your_results["TC"][benchmark]
        ref_tc = ref_results["TC"][benchmark]
        
        benchmark_passed = True
        for metric in key_metrics:
            total_tests += 1
            your_val = your_tc[metric]
            ref_val = ref_tc[metric]
            error = calculate_percentage_error(your_val, ref_val)
            within_tol = check_tolerance(your_val, ref_val)
            
            status = "✓ PASS" if within_tol else "✗ FAIL"
            if not within_tol:
                benchmark_passed = False
            
            print(f"  {metric}: {your_val:.1f} vs {ref_val:.1f} (error: {error:.2f}%) {status}")
        
        if benchmark_passed:
            passed_tests += 3  # 3 metrics per benchmark
        else:
            passed_tests += sum(1 for metric in key_metrics if check_tolerance(your_tc[metric], ref_tc[metric]))
    
    print("\n" + "=" * 80)
    print("FINAL GRADE CALCULATION:")
    print("-" * 50)
    
    # Calculate grade based on percentage of tests passed
    grade_percentage = (passed_tests / total_tests) * 100
    
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Grade percentage: {grade_percentage:.1f}%")
    
    if grade_percentage >= 95:
        print("🎉 EXCELLENT! Grade: A+ (100%)")
    elif grade_percentage >= 90:
        print("🎉 GREAT! Grade: A (95%)")
    elif grade_percentage >= 85:
        print("👍 GOOD! Grade: B+ (90%)")
    elif grade_percentage >= 80:
        print("👍 GOOD! Grade: B (85%)")
    else:
        print("❌ NEEDS IMPROVEMENT")
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS:")
    print("-" * 50)
    
    # Check if Task 2 shows different results from Task 1 (tensor core functionality)
    print("\nTENSOR CORE FUNCTIONALITY CHECK:")
    tensor_core_working = True
    
    for benchmark in ["gemm_half", "cnn_half", "ffn_half", "gpt2_half"]:
        cc_cycles = your_results["CC"][benchmark]["NUM_CYCLES"]
        tc_cycles = your_results["TC"][benchmark]["NUM_CYCLES"]
        cc_stalls = your_results["CC"][benchmark]["NUM_STALL_CYCLES"]
        tc_stalls = your_results["TC"][benchmark]["NUM_STALL_CYCLES"]
        
        cycles_diff = abs(tc_cycles - cc_cycles) / cc_cycles * 100
        stalls_diff = abs(tc_stalls - cc_stalls) / cc_stalls * 100
        
        if cycles_diff < 1.0 and stalls_diff < 1.0:
            print(f"  {benchmark}: ⚠️  WARNING - TC and CC results too similar!")
            tensor_core_working = False
        else:
            print(f"  {benchmark}: ✓ TC shows different results from CC (cycles: {cycles_diff:.1f}% diff, stalls: {stalls_diff:.1f}% diff)")
    
    if tensor_core_working:
        print("\n✅ TENSOR CORE FUNCTIONALITY: WORKING CORRECTLY")
    else:
        print("\n❌ TENSOR CORE FUNCTIONALITY: NEEDS IMPROVEMENT")

if __name__ == "__main__":
    analyze_results()
