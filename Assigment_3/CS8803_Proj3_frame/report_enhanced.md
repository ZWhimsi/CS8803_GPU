# GPU Warp Scheduling: GTO and CCWS Implementation

## Introduction

GPUs use warp scheduling to manage parallel threads. This report shows two algorithms: Greedy-Then-Oldest (GTO) and Cache-Conscious Wavefront Scheduling (CCWS). Both aim to improve GPU speed. One focuses on staying with the same warp. The other adjusts how many warps run based on cache behavior.

## Task 1: GTO Scheduler

### How It Works

GTO uses a greedy strategy. It keeps running the same warp until that warp needs to wait for memory. When it must switch, it picks the oldest waiting warp.

**Implementation steps:**

- Added `dispatch_time` to track warp age
- Used `c_last_scheduled_warp` pointer to remember the last warp
- Greedy rule: If last warp is still ready, run it again
- Oldest rule: When switching, find the warp with smallest `dispatch_time`
- Clear the pointer when warps suspend or finish

### Performance Results

| Benchmark        | RR Stalls | GTO Stalls | Change      |
| ---------------- | --------- | ---------- | ----------- |
| lavaMD_5         | 28,827    | 71,109     | -147% worse |
| nn_256k          | 663,693   | 576,007    | +13% better |
| backprop_8192    | 534,051   | 492,417    | +8% better  |
| crystal_q12      | 3,157,674 | 2,855,964  | +10% better |
| hotspot_r512h2i2 | 1,341,698 | 1,459,497  | -9% worse   |

**Why these results?** GTO helps when warps reuse cached data. It hurts when warps need different data or compute a lot without memory access. The results show that staying greedy works only for some workloads.

## Task 2: CCWS Scheduler

### How It Works

CCWS tries to reduce cache fights by limiting active warps. It uses a feedback system to detect lost locality.

**VTA (Victim Tag Array):**

- Each warp has its own VTA
- VTA stores tags of evicted cache lines
- When L1 evicts a line, save its tag in the VTA
- VTA holds 8 entries with LRU replacement

**LLS (Lost Locality Score):**

- On L1 miss, check if tag is in VTA
- If yes (VTA hit): warp lost locality
- Update score: `LLS = (VTA_Hits × 10 × Cutoff) / Total_Instructions`
- Scores decay by 1 each cycle (min = 100)

**Scheduling decision:**

- Calculate cutoff: `Active_Warps × 100`
- Sort warps by LLS (high to low)
- Add warps until cumulative score reaches cutoff
- Use Round Robin among selected warps

### Performance Results

| Benchmark        | RR Misses | GTO Misses | CCWS Misses | Change |
| ---------------- | --------- | ---------- | ----------- | ------ |
| lavaMD_5         | 9.99      | 3.01       | 9.87        | -1%    |
| nn_256k          | 93.71     | 91.13      | 93.71       | same   |
| backprop_8192    | 36.09     | 37.27      | 36.09       | same   |
| crystal_q12      | 75.47     | 75.47      | 75.47       | same   |
| hotspot_r512h2i2 | 180.92    | 182.05     | 180.92      | same   |

**Why minimal impact?** CCWS shows almost no difference from Round Robin. This is surprising. Perhaps these workloads don't create enough cache pressure. Or maybe the VTA size is too small to capture the pattern. Another possibility: the base score might be too high, allowing too many warps to stay active.

## Interesting Observations

### Workload Patterns

**Memory-heavy apps** (nn_256k, hotspot): These have many cache misses. GTO helps nn_256k a bit but hurts hotspot. Why the difference? Perhaps nn_256k has some reuse patterns while hotspot has random access.

**Compute-heavy apps** (lavaMD_5): Very few cache misses. GTO makes it worse. Why? Greedy scheduling reduces parallelism. With light memory use, we need more warps running to hide compute latency.

**Mixed apps** (backprop, crystal): Moderate cache misses. GTO helps both. These seem ideal for greedy scheduling—enough reuse to benefit from locality, enough memory activity to need switching.

### Algorithm Trade-offs

**GTO strengths:**

- Simple to implement
- Works well when warps reuse data
- Reduces warp switching overhead

**GTO weaknesses:**

- Reduces parallelism
- Can increase stalls if locality is poor
- No adaptation to changing patterns

**CCWS puzzle:**

- Complex feedback mechanism
- Minimal impact on these workloads
- Needs investigation: Is VTA too small? Base score too high? Wrong workloads?

### Questions for Future Work

Why does GTO hurt lavaMD_5 so much? Is it purely parallelism loss, or something else?

Why does CCWS barely change anything? Should we test with:

- Larger VTA (16 or 32 entries)?
- Lower base score (50 instead of 100)?
- Different decay rate?
- Workloads with more cache pressure?

Could we combine GTO and CCWS? Use GTO for warps with high hit rates and RR for others?

## Conclusion

Warp scheduling is not one-size-fits-all. GTO improves performance for some workloads (up to 13%) but hurts others (up to 147% worse). CCWS shows minimal impact, raising questions about its effectiveness for these specific benchmarks.

The key insight: scheduling algorithms must match workload characteristics. Greedy works when locality exists. Dynamic throttling needs sufficient cache pressure to show benefits.

Both implementations work correctly (matching reference results exactly). The performance differences reveal the complexity of GPU optimization. What helps one application can hurt another. This makes GPU scheduling an ongoing research challenge.
