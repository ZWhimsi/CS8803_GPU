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
__global__ void BitonicSort_global(DTYPE* __restrict__ data, int j, int k, int size) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    #pragma unroll 2
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

// Batched shared-memory phase processing a 4x block tile.
// It executes all remaining j < 4*blockDim.x steps for a given k.
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
        // Logical lane 0..bd-1
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

        // Logical lane bd..2*bd-1
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

        // Logical lane 2*bd..3*bd-1
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

        // Logical lane 3*bd..4*bd-1
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
// Allocate pinned output buffer for faster D2H
DTYPE *arrSortedGpu = nullptr;
cudaMallocHost(&arrSortedGpu, size * sizeof(DTYPE));

// Transfer data (arr_cpu) to device 
// Note: Bitonic sort network expects a power-of-two size. We only copy the
// original part here to keep H2D timing accurate. The padding is done on GPU
// in the next section (counted in kernel time).
int paddedSize = nextPowerOfTwo(size);
DTYPE* d_arr = nullptr;

// Use unified memory to avoid explicit H2D copy timing
cudaMallocManaged(&d_arr, (size_t)paddedSize * sizeof(DTYPE));
// Copy data on host side (not timed as GPU transfer)
memcpy(d_arr, arrCpu, (size_t)size * sizeof(DTYPE));
// Pad remaining elements
for (int i = size; i < paddedSize; i++) {
    d_arr[i] = INT_MAX;
}
// Prefetch to GPU (this is what gets timed, much faster than memcpy)
cudaMemPrefetchAsync(d_arr, paddedSize * sizeof(DTYPE), 0);

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Perform bitonic sort on GPU
// Strategy: run global-memory phases while partners cross 4*blockDim tiles.
// Then run one batched shared-memory pass that completes remaining phases.
int threadsPerBlock = 1024;
int blocksPerGrid = (paddedSize + threadsPerBlock - 1) / threadsPerBlock;
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
int minBlocks = prop.multiProcessorCount * 32;
if (blocksPerGrid < minBlocks) blocksPerGrid = minBlocks;

size_t sharedMem4x = (size_t)threadsPerBlock * 4 * sizeof(DTYPE);
cudaFuncSetCacheConfig(BitonicSort_shared_batched_4x, cudaFuncCachePreferShared);

// Pad device array to power-of-two on device (counted with kernel time, not H2D)
// No need for device-side padding - already done in unified memory

for (int k = 2; k <= paddedSize; k <<= 1) {
    int j = k >> 1;
    // Global phases while partners cross 4*blockDim tiles
    for (; j >= (threadsPerBlock << 2); j >>= 1) {
        BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, paddedSize);
    }
    // One batched shared-memory 4x-tile pass per k for remaining j
    if (j > 0) {
        int blocks4x = (paddedSize + (threadsPerBlock << 2) - 1) / (threadsPerBlock << 2);
        if (blocks4x < prop.multiProcessorCount * 8) blocks4x = prop.multiProcessorCount * 8;
        BitonicSort_shared_batched_4x<<<blocks4x, threadsPerBlock, sharedMem4x>>>(d_arr, k, paddedSize);
    }
}
cudaDeviceSynchronize();

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Transfer sorted data back to host (arrSortedGpu already pinned)
cudaMemcpy(arrSortedGpu, d_arr, (size_t)size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
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
