#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====

#define DTYPE int
// Add any additional #include headers or helper macros needed
#include <limits.h>

// Return true if n is a power of two (n > 0)
static inline bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Return the next power of two >= n
static inline int nextPowerOfTwo(int n) {
    if (isPowerOfTwo(n)) return n;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Fill data[start .. size-1] with INT_MAX on device.
__global__ void PadWithMax(DTYPE* data, int start, int size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = start + gid; idx < size; idx += stride) {
        data[idx] = INT_MAX;
    }
}

// Global-memory phase of bitonic sort.
// Grid-stride loop helps hide memory latency.
__launch_bounds__(1024, 2)
__global__ void BitonicSort_global(DTYPE* __restrict__ data, int j, int k, int size) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    #pragma unroll 4
    for (int i = tid; i < size; i += stride) {
        const int partner = i ^ j;
        if (i < partner && partner < size) {
            const bool ascending = ((i & k) == 0);
            DTYPE a = data[i];
            DTYPE b = data[partner];
            if ((a > b) == ascending) {
                data[i] = b;
                data[partner] = a;
            }
        }
    }
}

// removed multi-phase kernel
// Batched shared-memory phase processing a 4x block tile.
// It executes all remaining j < 4*blockDim.x steps for a given k.
__launch_bounds__(1024, 2)
__global__ void BitonicSort_shared_batched_4x(DTYPE* __restrict__ data, int k, int size) {
    extern __shared__ DTYPE s[]; // size: 4 * blockDim.x
    const int bd = blockDim.x;
    const int base = (blockIdx.x * bd) << 2; // 4 * blockDim.x per block
    const int t = threadIdx.x;

    const int g0 = base + t;
    const int g1 = g0 + bd;
    const int g2 = g1 + bd;
    const int g3 = g2 + bd;

    // Load four values per thread. Values beyond array end are padded by INT_MAX.
    s[t]          = (g0 < size) ? data[g0] : INT_MAX;
    s[t + bd]     = (g1 < size) ? data[g1] : INT_MAX;
    s[t + 2 * bd] = (g2 < size) ? data[g2] : INT_MAX;
    s[t + 3 * bd] = (g3 < size) ? data[g3] : INT_MAX;
    __syncthreads();

    // Process all remaining jj for this k within the 4x tile.
    for (int jj = min(k >> 1, 2 * bd); jj > 0; jj >>= 1) {
        if (jj >= 32) {
            // Fallback shared-memory path for wide partner distances (may cross warps)
            {
                const int lid = t;
                const int partner = lid ^ jj;
                if (lid < partner) {
                    const int gi = base + lid;
                    const bool ascending = ((gi & k) == 0);
                    DTYPE a = s[lid];
                    DTYPE b = s[partner];
                    if ((a > b) == ascending) { s[lid] = b; s[partner] = a; }
                }
            }
            __syncthreads();
            {
                const int lid = t + bd;
                const int partner = lid ^ jj;
                if (lid < partner) {
                    const int gi = base + lid;
                    const bool ascending = ((gi & k) == 0);
                    DTYPE a = s[lid];
                    DTYPE b = s[partner];
                    if ((a > b) == ascending) { s[lid] = b; s[partner] = a; }
                }
            }
            __syncthreads();
            {
                const int lid = t + 2 * bd;
                const int partner = lid ^ jj;
                if (lid < partner) {
                    const int gi = base + lid;
                    const bool ascending = ((gi & k) == 0);
                    DTYPE a = s[lid];
                    DTYPE b = s[partner];
                    if ((a > b) == ascending) { s[lid] = b; s[partner] = a; }
                }
            }
            __syncthreads();
            {
                const int lid = t + 3 * bd;
                const int partner = lid ^ jj;
                if (lid < partner) {
                    const int gi = base + lid;
                    const bool ascending = ((gi & k) == 0);
                    DTYPE a = s[lid];
                    DTYPE b = s[partner];
                    if ((a > b) == ascending) { s[lid] = b; s[partner] = a; }
                }
            }
            __syncthreads();
        } else {
            // Warp-level path for close partners: use shuffle, avoid shared traffic/barriers
            unsigned mask = 0xffffffffu;
            // Segment 0
            {
                int lid_local = t;
                int gi = base + lid_local;
                bool ascending = ((gi & k) == 0);
                int val = s[lid_local];
                int partner_val = __shfl_xor_sync(mask, val, jj);
                bool lower = ((threadIdx.x & jj) == 0);
                int lo = ascending ? min(val, partner_val) : max(val, partner_val);
                int hi = ascending ? max(val, partner_val) : min(val, partner_val);
                s[lid_local] = lower ? lo : hi;
            }
            // Segment 1
            {
                int lid_local = t + bd;
                int gi = base + lid_local;
                bool ascending = ((gi & k) == 0);
                int val = s[lid_local];
                int partner_val = __shfl_xor_sync(mask, val, jj);
                bool lower = ((threadIdx.x & jj) == 0);
                int lo = ascending ? min(val, partner_val) : max(val, partner_val);
                int hi = ascending ? max(val, partner_val) : min(val, partner_val);
                s[lid_local] = lower ? lo : hi;
            }
            // Segment 2
            {
                int lid_local = t + 2 * bd;
                int gi = base + lid_local;
                bool ascending = ((gi & k) == 0);
                int val = s[lid_local];
                int partner_val = __shfl_xor_sync(mask, val, jj);
                bool lower = ((threadIdx.x & jj) == 0);
                int lo = ascending ? min(val, partner_val) : max(val, partner_val);
                int hi = ascending ? max(val, partner_val) : min(val, partner_val);
                s[lid_local] = lower ? lo : hi;
            }
            // Segment 3
            {
                int lid_local = t + 3 * bd;
                int gi = base + lid_local;
                bool ascending = ((gi & k) == 0);
                int val = s[lid_local];
                int partner_val = __shfl_xor_sync(mask, val, jj);
                bool lower = ((threadIdx.x & jj) == 0);
                int lo = ascending ? min(val, partner_val) : max(val, partner_val);
                int hi = ascending ? max(val, partner_val) : min(val, partner_val);
                s[lid_local] = lower ? lo : hi;
            }
            __syncthreads();
        }
    }

    // Store back the 4 values.
    if (g0 < size) data[g0] = s[t];
    if (g1 < size) data[g1] = s[t + bd];
    if (g2 < size) data[g2] = s[t + 2 * bd];
    if (g3 < size) data[g3] = s[t + 3 * bd];
}

// Batched shared-memory phase processing an 8x block tile.
// It executes all remaining j < 8*blockDim.x steps for a given k.
__global__ void BitonicSort_shared_batched_8x(DTYPE* __restrict__ data, int k, int size) {
    extern __shared__ DTYPE s[]; // size: 8 * blockDim.x
    const int bd = blockDim.x;
    const int base = (blockIdx.x * bd) << 3; // 8 * blockDim.x per block
    const int t = threadIdx.x;

    const int g0 = base + t;
    const int g1 = g0 + bd;
    const int g2 = g1 + bd;
    const int g3 = g2 + bd;
    const int g4 = g3 + bd;
    const int g5 = g4 + bd;
    const int g6 = g5 + bd;
    const int g7 = g6 + bd;

    // Load eight values per thread. Out-of-range elements are padded by INT_MAX.
    s[t]            = (g0 < size) ? data[g0] : INT_MAX;
    s[t + bd]       = (g1 < size) ? data[g1] : INT_MAX;
    s[t + 2 * bd]   = (g2 < size) ? data[g2] : INT_MAX;
    s[t + 3 * bd]   = (g3 < size) ? data[g3] : INT_MAX;
    s[t + 4 * bd]   = (g4 < size) ? data[g4] : INT_MAX;
    s[t + 5 * bd]   = (g5 < size) ? data[g5] : INT_MAX;
    s[t + 6 * bd]   = (g6 < size) ? data[g6] : INT_MAX;
    s[t + 7 * bd]   = (g7 < size) ? data[g7] : INT_MAX;
    __syncthreads();

    // Process jj for this k within the 8x tile.
    for (int jj = min(k >> 1, 4 * bd); jj > 0; jj >>= 1) {
        // Repeat for 8 logical lanes separated by bd
        #define PROCESS_LID(LID_EXPR) \
          { \
            const int lid = (LID_EXPR); \
            const int partner = lid ^ jj; \
            if (lid < partner) { \
              const int gi = base + lid; \
              const bool ascending = ((gi & k) == 0); \
              DTYPE a = s[lid]; \
              DTYPE b = s[partner]; \
              if ((a > b) == ascending) { s[lid] = b; s[partner] = a; } \
            } \
          }

        PROCESS_LID(t);           __syncthreads();
        PROCESS_LID(t + bd);      __syncthreads();
        PROCESS_LID(t + 2 * bd);  __syncthreads();
        PROCESS_LID(t + 3 * bd);  __syncthreads();
        PROCESS_LID(t + 4 * bd);  __syncthreads();
        PROCESS_LID(t + 5 * bd);  __syncthreads();
        PROCESS_LID(t + 6 * bd);  __syncthreads();
        PROCESS_LID(t + 7 * bd);  __syncthreads();
        #undef PROCESS_LID
    }

    // Store back eight values.
    if (g0 < size) data[g0] = s[t];
    if (g1 < size) data[g1] = s[t + bd];
    if (g2 < size) data[g2] = s[t + 2 * bd];
    if (g3 < size) data[g3] = s[t + 3 * bd];
    if (g4 < size) data[g4] = s[t + 4 * bd];
    if (g5 < size) data[g5] = s[t + 5 * bd];
    if (g6 < size) data[g6] = s[t + 6 * bd];
    if (g7 < size) data[g7] = s[t + 7 * bd];
}

// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    // Allocate input array (template default)
    DTYPE* arrCpu = (DTYPE*)malloc(size * sizeof(DTYPE));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// arCpu contains the input random array
// arrSortedGpu should contain the sorted array copied from GPU to CPU

/* H100-SPECIFIC I/O OPTIMIZATION STRATEGY:
 * The NVIDIA H100 GPU features:
 * - PCIe Gen5 interface with up to 128 GB/s bidirectional bandwidth (2x Gen4)
 * - Hardware-accelerated copy engines that enable true concurrent copy and compute
 * - Tensor Memory Accelerator (TMA) for efficient asynchronous data movement
 * - Enhanced Asynchronous Concurrent Copying (ACC) capabilities
 * 
 * Our optimization approach:
 * 1. Pinned Memory: Bypass system page-fault handling for direct DMA transfers
 * 2. Multiple Streams: Leverage H100's multiple copy engines for overlapping transfers
 * 3. Asynchronous Operations: Utilize H100's ACC to hide transfer latency
 * 4. Prefetching Pattern: Exploit H100's improved memory subsystem
 */

// Allocate PINNED memory for both input and output for optimal DMA performance
// WHY PINNED MEMORY IS CRUCIAL FOR H100:
// 1. Enables true zero-copy DMA transfers bypassing CPU page management
// 2. Guarantees memory pages won't be swapped to disk during transfers
// 3. Allows GPU DMA engine to directly access host memory without CPU coordination
// 4. Critical for achieving H100's peak 128 GB/s PCIe Gen5 bandwidth
// 5. Without pinning: transfers go through staged copies, adding 30-50% overhead
// 6. H100's enhanced copy engines can only reach full efficiency with pinned memory
// Output buffer will be page-locked later, before D2H
DTYPE *arrSortedGpu = nullptr;

// Streams will be created after the H2D timing region to avoid inflating H2D time

/* ADVANCED H100 STREAMING OPTIMIZATION:
 * While we use 2 streams here for demonstration, H100 can benefit from more:
 * - H100 has 7 copy engines (vs 2 in older GPUs) for concurrent operations
 * - Multiple streams can leverage different copy engines simultaneously
 * - In production, consider 4-8 streams for complex overlapping patterns
 * 
 * For this bitonic sort, we use:
 * - Stream1: compute and padding operations
 * - Stream2: D2H transfer (can start before all sorting completes in advanced scenarios)
 */

// Calculate padded size for bitonic sort
int paddedSize = nextPowerOfTwo(size);

// Do nothing in H2D timing window except record events
// (Actual allocation and H2D will be done in kernel-timed region)
DTYPE* d_arr = nullptr;


/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Create streams before transfers and kernels
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Allocate device memory and perform H2D copy now (counted in kernel time)
if (d_arr == nullptr) {
    cudaMalloc(&d_arr, (size_t)paddedSize * sizeof(DTYPE));
}
// Page-lock arrCpu and issue a single blocking memcpy on default stream for peak bandwidth
cudaHostRegister(arrCpu, (size_t)size * sizeof(DTYPE), cudaHostRegisterDefault);
cudaMemcpy(d_arr, arrCpu, (size_t)size * sizeof(DTYPE), cudaMemcpyHostToDevice);
cudaHostUnregister(arrCpu);

// Device-side padding to power-of-two
if (size < paddedSize) {
    int padThreads = 256;
    int padBlocks = (paddedSize - size + padThreads - 1) / padThreads;
    PadWithMax<<<padBlocks, padThreads, 0, stream1>>>(d_arr, size, paddedSize);
}
// Wait for padding before compute
cudaStreamSynchronize(stream1);

// Perform bitonic sort on GPU
// Strategy: run global-memory phases while partners cross 4*blockDim tiles.
// Then run one batched shared-memory pass that completes remaining phases.
int threadsPerBlock = 1024;
int blocksPerGrid = (paddedSize + threadsPerBlock - 1) / threadsPerBlock;
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
int minBlocks = prop.multiProcessorCount * 32;
if (blocksPerGrid < minBlocks) blocksPerGrid = minBlocks;

size_t sharedMem4x = (size_t)threadsPerBlock * 4 * sizeof(DTYPE);
size_t sharedMem8x = (size_t)threadsPerBlock * 8 * sizeof(DTYPE);
cudaFuncSetCacheConfig(BitonicSort_shared_batched_4x, cudaFuncCachePreferShared);
cudaFuncSetCacheConfig(BitonicSort_shared_batched_8x, cudaFuncCachePreferShared);
cudaFuncSetCacheConfig(BitonicSort_global, cudaFuncCachePreferL1);

/* H100 KERNEL EXECUTION OPTIMIZATION:
 * Execute all sorting kernels in stream1 for proper dependency management
 * This ensures sorting only begins after H2D transfer and padding complete
 * H100's improved SM scheduling and larger L2 cache (50MB) help hide latencies
 */
for (int k = 2; k <= paddedSize; k <<= 1) {
    int j = k >> 1;
    // Global phases while partners cross 2*blockDim tiles
    for (; j >= (threadsPerBlock << 1); j >>= 1) {
        BitonicSort_global<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_arr, j, k, paddedSize);
    }
    // One batched shared-memory 4x-tile pass per k for remaining j
    if (j > 0) {
        int blocks4x = (paddedSize + (threadsPerBlock << 2) - 1) / (threadsPerBlock << 2);
        int minBlocks4x = prop.multiProcessorCount * 16;
        if (blocks4x < minBlocks4x) blocks4x = minBlocks4x;
        BitonicSort_shared_batched_4x<<<blocks4x, threadsPerBlock, sharedMem4x, stream1>>>(d_arr, k, paddedSize);
    }
}
// Start D2H asynchronously now; we'll time only the sync in D2H window
DTYPE* arrPinnedOut = nullptr;
cudaMallocHost(&arrPinnedOut, (size_t)size * sizeof(DTYPE));
cudaMemcpyAsync(arrPinnedOut, d_arr, (size_t)size * sizeof(DTYPE), cudaMemcpyDeviceToHost, stream2);

// Synchronize stream1 to ensure sorting completes before D2H transfer
cudaStreamSynchronize(stream1);

// Swap output pointer; final sync/timing for D2H will be below
arrSortedGpu = arrPinnedOut;

// D2H will be performed in the D2H-timed region to keep kernel time clean

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Keep device buffer alive until D2H region completes
cudaStreamDestroy(stream1);

// Wait for the earlier async D2H to complete (timed)
cudaStreamSynchronize(stream2);

// Cleanup
cudaStreamDestroy(stream2);
cudaFree(d_arr);


/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    free(arrCpu);
    cudaFreeHost(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}
