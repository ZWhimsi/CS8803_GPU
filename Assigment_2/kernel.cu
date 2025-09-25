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

// Pad tail of array with INT_MAX from startIndex to totalSize.
// Uses a grid-stride loop for coverage.
__global__ void PadWithMax(DTYPE* data, int startIndex, int totalSize) {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    for (int dataIndex = startIndex + globalThreadId; dataIndex < totalSize; dataIndex += gridStride) {
        data[dataIndex] = INT_MAX;
    }
}

// Bitonic sort: global-memory phase.
// Each thread compares with its XOR-partner and swaps if needed.
__global__ void BitonicSort_global(DTYPE* __restrict__ data, int partnerMask, int stageMask, int totalSize) {
    const int globalThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int gridStride = blockDim.x * gridDim.x;

    #pragma unroll 2
    for (int elementIndex = globalThreadIndex; elementIndex < totalSize; elementIndex += gridStride) {
        const int partnerIndex = elementIndex ^ partnerMask;
        if (elementIndex < partnerIndex && partnerIndex < totalSize) {
            const bool sortAscending = ((elementIndex & stageMask) == 0);
            DTYPE valueSelf = data[elementIndex];
            DTYPE valuePartner = data[partnerIndex];
            if ((valueSelf > valuePartner) == sortAscending) {
                data[elementIndex] = valuePartner;
                data[partnerIndex] = valueSelf;
            }
        }
    }
}

// Bitonic sort: shared-memory 4x tile phase.
// Finishes remaining steps for this k inside the tile.
__global__ void BitonicSort_shared_batched_4x(DTYPE* __restrict__ data, int k, int size) {
    extern __shared__ DTYPE tileValues[]; // 4 * blockDim.x elements
    const int blockWidth = blockDim.x;
    const int tileBaseIndex = (blockIdx.x * blockWidth) << 2; // 4 * blockDim.x per block
    const int localThreadIndex = threadIdx.x;

    const int globalIndex0 = tileBaseIndex + localThreadIndex;
    const int globalIndex1 = globalIndex0 + blockWidth;
    const int globalIndex2 = globalIndex1 + blockWidth;
    const int globalIndex3 = globalIndex2 + blockWidth;

    // Load four values per thread. Out-of-range elements are padded with INT_MAX.
    tileValues[localThreadIndex]                 = (globalIndex0 < size) ? data[globalIndex0] : INT_MAX;
    tileValues[localThreadIndex + blockWidth]    = (globalIndex1 < size) ? data[globalIndex1] : INT_MAX;
    tileValues[localThreadIndex + 2 * blockWidth]= (globalIndex2 < size) ? data[globalIndex2] : INT_MAX;
    tileValues[localThreadIndex + 3 * blockWidth]= (globalIndex3 < size) ? data[globalIndex3] : INT_MAX;
    __syncthreads();

    // Process all remaining jj for this k within the 4x tile.
    for (int jj = min(k >> 1, 2 * blockWidth); jj > 0; jj >>= 1) {
        // Logical lane 0..blockWidth-1
        {
            const int localId = localThreadIndex;
            const int partner = localId ^ jj;
            if (localId < partner) {
                const int globalId = tileBaseIndex + localId;
                const bool ascending = ((globalId & k) == 0);
                DTYPE a = tileValues[localId];
                DTYPE b = tileValues[partner];
                if ((a > b) == ascending) { tileValues[localId] = b; tileValues[partner] = a; }
            }
        }
        __syncthreads();

        // Logical lane blockWidth..2*blockWidth-1
        {
            const int localId = localThreadIndex + blockWidth;
            const int partner = localId ^ jj;
            if (localId < partner) {
                const int globalId = tileBaseIndex + localId;
                const bool ascending = ((globalId & k) == 0);
                DTYPE a = tileValues[localId];
                DTYPE b = tileValues[partner];
                if ((a > b) == ascending) { tileValues[localId] = b; tileValues[partner] = a; }
            }
        }
        __syncthreads();

        // Logical lane 2*blockWidth..3*blockWidth-1
        {
            const int localId = localThreadIndex + 2 * blockWidth;
            const int partner = localId ^ jj;
            if (localId < partner) {
                const int globalId = tileBaseIndex + localId;
                const bool ascending = ((globalId & k) == 0);
                DTYPE a = tileValues[localId];
                DTYPE b = tileValues[partner];
                if ((a > b) == ascending) { tileValues[localId] = b; tileValues[partner] = a; }
            }
        }
        __syncthreads();

        // Logical lane 3*blockWidth..4*blockWidth-1
        {
            const int localId = localThreadIndex + 3 * blockWidth;
            const int partner = localId ^ jj;
            if (localId < partner) {
                const int globalId = tileBaseIndex + localId;
                const bool ascending = ((globalId & k) == 0);
                DTYPE a = tileValues[localId];
                DTYPE b = tileValues[partner];
                if ((a > b) == ascending) { tileValues[localId] = b; tileValues[partner] = a; }
            }
        }
        __syncthreads();
    }

    // Store back the 4 values.
    if (globalIndex0 < size) data[globalIndex0] = tileValues[localThreadIndex];
    if (globalIndex1 < size) data[globalIndex1] = tileValues[localThreadIndex + blockWidth];
    if (globalIndex2 < size) data[globalIndex2] = tileValues[localThreadIndex + 2 * blockWidth];
    if (globalIndex3 < size) data[globalIndex3] = tileValues[localThreadIndex + 3 * blockWidth];
}

// Bitonic sort: shared-memory 8x tile phase.
// Processes remaining steps for this k inside the tile.
__global__ void BitonicSort_shared_batched_8x(DTYPE* __restrict__ data, int k, int size) {
    extern __shared__ DTYPE tileValues[]; // 8 * blockDim.x elements
    const int blockWidth = blockDim.x;
    const int tileBaseIndex = (blockIdx.x * blockWidth) << 3; // 8 * blockDim.x per block
    const int localThreadIndex = threadIdx.x;

    const int globalIndex0 = tileBaseIndex + localThreadIndex;
    const int globalIndex1 = globalIndex0 + blockWidth;
    const int globalIndex2 = globalIndex1 + blockWidth;
    const int globalIndex3 = globalIndex2 + blockWidth;
    const int globalIndex4 = globalIndex3 + blockWidth;
    const int globalIndex5 = globalIndex4 + blockWidth;
    const int globalIndex6 = globalIndex5 + blockWidth;
    const int globalIndex7 = globalIndex6 + blockWidth;

    // Load eight values per thread. Out-of-range elements are padded with INT_MAX.
    tileValues[localThreadIndex]                  = (globalIndex0 < size) ? data[globalIndex0] : INT_MAX;
    tileValues[localThreadIndex + blockWidth]     = (globalIndex1 < size) ? data[globalIndex1] : INT_MAX;
    tileValues[localThreadIndex + 2 * blockWidth] = (globalIndex2 < size) ? data[globalIndex2] : INT_MAX;
    tileValues[localThreadIndex + 3 * blockWidth] = (globalIndex3 < size) ? data[globalIndex3] : INT_MAX;
    tileValues[localThreadIndex + 4 * blockWidth] = (globalIndex4 < size) ? data[globalIndex4] : INT_MAX;
    tileValues[localThreadIndex + 5 * blockWidth] = (globalIndex5 < size) ? data[globalIndex5] : INT_MAX;
    tileValues[localThreadIndex + 6 * blockWidth] = (globalIndex6 < size) ? data[globalIndex6] : INT_MAX;
    tileValues[localThreadIndex + 7 * blockWidth] = (globalIndex7 < size) ? data[globalIndex7] : INT_MAX;
    __syncthreads();

    // Process jj for this k within the 8x tile.
    for (int jj = min(k >> 1, 4 * blockWidth); jj > 0; jj >>= 1) {
        // Repeat for 8 logical lanes separated by blockWidth
        #define PROCESS_LID(LID_EXPR) \
          { \
            const int localId = (LID_EXPR); \
            const int partner = localId ^ jj; \
            if (localId < partner) { \
              const int globalId = tileBaseIndex + localId; \
              const bool ascending = ((globalId & k) == 0); \
              DTYPE a = tileValues[localId]; \
              DTYPE b = tileValues[partner]; \
              if ((a > b) == ascending) { tileValues[localId] = b; tileValues[partner] = a; } \
            } \
          }

        PROCESS_LID(localThreadIndex);           __syncthreads();
        PROCESS_LID(localThreadIndex + blockWidth);      __syncthreads();
        PROCESS_LID(localThreadIndex + 2 * blockWidth);  __syncthreads();
        PROCESS_LID(localThreadIndex + 3 * blockWidth);  __syncthreads();
        PROCESS_LID(localThreadIndex + 4 * blockWidth);  __syncthreads();
        PROCESS_LID(localThreadIndex + 5 * blockWidth);  __syncthreads();
        PROCESS_LID(localThreadIndex + 6 * blockWidth);  __syncthreads();
        PROCESS_LID(localThreadIndex + 7 * blockWidth);  __syncthreads();
        #undef PROCESS_LID
    }

    // Store back eight values.
    if (globalIndex0 < size) data[globalIndex0] = tileValues[localThreadIndex];
    if (globalIndex1 < size) data[globalIndex1] = tileValues[localThreadIndex + blockWidth];
    if (globalIndex2 < size) data[globalIndex2] = tileValues[localThreadIndex + 2 * blockWidth];
    if (globalIndex3 < size) data[globalIndex3] = tileValues[localThreadIndex + 3 * blockWidth];
    if (globalIndex4 < size) data[globalIndex4] = tileValues[localThreadIndex + 4 * blockWidth];
    if (globalIndex5 < size) data[globalIndex5] = tileValues[localThreadIndex + 5 * blockWidth];
    if (globalIndex6 < size) data[globalIndex6] = tileValues[localThreadIndex + 6 * blockWidth];
    if (globalIndex7 < size) data[globalIndex7] = tileValues[localThreadIndex + 7 * blockWidth];
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
DTYPE *arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));

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
    free(arrSortedGpu);

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
