#!/usr/bin/env python3

# Your results vs Reference results for Task 2 (TC) - NUM_STALL_CYCLES
benchmarks = ["gemm_float", "gemm_half", "cnn_float", "cnn_half", "ffn_float", "ffn_half", "gpt2_float", "gpt2_half"]

your_tc = {
    "gemm_float": 362254.0,
    "gemm_half": 281982.0,
    "cnn_float": 7049370.0,
    "cnn_half": 13665083.0,
    "ffn_float": 42174435.0,
    "ffn_half": 9680713.0,
    "gpt2_float": 34780934.0,
    "gpt2_half": 24877254.0
}

ref_tc = {
    "gemm_float": 362254.0,
    "gemm_half": 283688.0,
    "cnn_float": 7049370.0,
    "cnn_half": 14074545.0,
    "ffn_float": 42174435.0,
    "ffn_half": 10188086.0,
    "gpt2_float": 34780934.0,
    "gpt2_half": 26266192.0
}

print("TASK 2 (TC) DIFFERENCES ANALYSIS:")
print("=" * 60)
print("Benchmark        | Your Value | Ref Value | Difference | Error %")
print("-" * 60)

for benchmark in benchmarks:
    your_val = your_tc[benchmark]
    ref_val = ref_tc[benchmark]
    diff = your_val - ref_val
    error = abs(diff / ref_val) * 100 if ref_val != 0 else 0
    
    status = "EXACT" if diff == 0 else f"{diff:+.0f}"
    print(f"{benchmark:15} | {your_val:10.0f} | {ref_val:9.0f} | {status:>10} | {error:6.2f}%")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("-" * 60)

# Find patterns in differences
exact_matches = [b for b in benchmarks if your_tc[b] == ref_tc[b]]
differences = [b for b in benchmarks if your_tc[b] != ref_tc[b]]

print(f"Exact matches: {len(exact_matches)}/8")
print(f"Differences: {len(differences)}/8")

if differences:
    print(f"\nBenchmarks with differences: {differences}")
    
    # Check if differences are consistent
    your_diffs = [your_tc[b] - ref_tc[b] for b in differences]
    print(f"Difference values: {your_diffs}")
    
    # Check if all differences are in same direction
    all_positive = all(d > 0 for d in your_diffs)
    all_negative = all(d < 0 for d in your_diffs)
    
    if all_positive:
        print("🔍 PATTERN: All differences are positive (your values are higher)")
        print("   This suggests your implementation might be:")
        print("   - Adding extra stall cycles somewhere")
        print("   - Not removing completed instructions properly")
        print("   - Buffer management issue")
    elif all_negative:
        print("🔍 PATTERN: All differences are negative (your values are lower)")
        print("   This suggests your implementation might be:")
        print("   - Missing some stall cycles")
        print("   - Removing instructions too early")
        print("   - Buffer capacity issue")
    else:
        print("🔍 PATTERN: Mixed positive and negative differences")
        print("   This suggests random variations, possibly:")
        print("   - Timing differences in instruction completion")
        print("   - Race conditions in scheduling")
        print("   - Order-dependent operations")

print("\n" + "=" * 60)
print("COMMON GPU IMPLEMENTATION ISSUES TO CHECK:")
print("-" * 60)
print("1. Instruction completion timing:")
print("   - Are you removing completed instructions at the right cycle?")
print("   - Is the completion timestamp calculation correct?")
print()
print("2. Buffer management:")
print("   - Is the execution buffer size correct?")
print("   - Are you checking buffer fullness correctly?")
print()
print("3. Stall cycle counting:")
print("   - Are you incrementing stall_cycles at the right time?")
print("   - Is the stall condition logic correct?")
print()
print("4. Tensor core latency:")
print("   - Is the 64-cycle latency applied correctly to H* opcodes?")
print("   - Are other instructions still using 1-cycle latency?")
print()
print("5. Execution width:")
print("   - Is the execution width of 8 being used correctly?")
print("   - Are you checking buffer capacity against execution_width?")
