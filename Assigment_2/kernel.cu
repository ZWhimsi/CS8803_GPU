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
static inline bool Pow_two_checker(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Return the next power of two >= n
static inline int next_pow(int n) {
    if (Pow_two_checker(n)) return n;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Pad tail of array with INT_MAX from startIndex to totalSize.
// Uses a grid-stride loop for coverage.
__global__ void PadWithMax(DTYPE* data, int start_ind, int size) {
    const int glob_ind = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    for (int data_ind = start_ind + glob_ind; data_ind < size; data_ind += grid_stride) {
        data[data_ind] = INT_MAX;
    }
}

// Bitonic sort: global-memory phase.
// Each thread compares with its XOR-partner and swaps if needed.
__global__ void BitonicSort_global(DTYPE* __restrict__ data, int partnerMask, int stageMask, int size) {
    const int glob_thread_ind = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;

    #pragma unroll 2
    for (int elmnt_ind = glob_thread_ind; elmnt_ind < size; elmnt_ind += grid_stride) {
        const int partnerIndex = elmnt_ind ^ partnerMask;
        if (elmnt_ind < partnerIndex && partnerIndex < size) {
            const bool ascending = ((elmnt_ind & stageMask) == 0);
            DTYPE a = data[elmnt_ind];
            DTYPE b = data[partnerIndex];
            if ((a > b) == ascending) {
                data[elmnt_ind] = b;
                data[partnerIndex] = a;
            }
        }
    }
}

// Bitonic sort: shared-memory 4x tile phase.
// Finishes remaining steps for this k inside the tile.
__global__ void BitonicSort_shared_batched_4x(DTYPE* __restrict__ data, int k, int size) {
    extern __shared__ DTYPE tile[]; // 4 * blockDim.x
    const int W = blockDim.x;
    const int base = (blockIdx.x * W) << 2; // four stripes
    const int t = threadIdx.x;

    const int g0 = base + t;
    const int g1 = g0 + W;
    const int g2 = g1 + W;
    const int g3 = g2 + W;

    tile[t]       = (g0 < size) ? data[g0] : INT_MAX;
    tile[t + W]   = (g1 < size) ? data[g1] : INT_MAX;
    tile[t + 2*W] = (g2 < size) ? data[g2] : INT_MAX;
    tile[t + 3*W] = (g3 < size) ? data[g3] : INT_MAX;
    __syncthreads();

    for (int jj = min(k >> 1, 2 * W); jj > 0; jj >>= 1) {
        {
            const int lid = t;
            const int partner = lid ^ jj;
            if (lid < partner) {
                const int gi = base + lid;
                const bool asc = ((gi & k) == 0);
                DTYPE a = tile[lid];
                DTYPE b = tile[partner];
                if ((a > b) == asc) { tile[lid] = b; tile[partner] = a; }
            }
        }
        __syncthreads();
        {
            const int lid = t + W;
            const int partner = lid ^ jj;
            if (lid < partner) {
                const int gi = base + lid;
                const bool asc = ((gi & k) == 0);
                DTYPE a = tile[lid];
                DTYPE b = tile[partner];
                if ((a > b) == asc) { tile[lid] = b; tile[partner] = a; }
            }
        }
        __syncthreads();
        {
            const int lid = t + 2*W;
            const int partner = lid ^ jj;
            if (lid < partner) {
                const int gi = base + lid;
                const bool asc = ((gi & k) == 0);
                DTYPE a = tile[lid];
                DTYPE b = tile[partner];
                if ((a > b) == asc) { tile[lid] = b; tile[partner] = a; }
            }
        }
        __syncthreads();
        {
            const int lid = t + 3*W;
            const int partner = lid ^ jj;
            if (lid < partner) {
                const int gi = base + lid;
                const bool asc = ((gi & k) == 0);
                DTYPE a = tile[lid];
                DTYPE b = tile[partner];
                if ((a > b) == asc) { tile[lid] = b; tile[partner] = a; }
            }
        }
        __syncthreads();
    }

    if (g0 < size) data[g0] = tile[t];
    if (g1 < size) data[g1] = tile[t + W];
    if (g2 < size) data[g2] = tile[t + 2*W];
    if (g3 < size) data[g3] = tile[t + 3*W];
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
int padded_size = next_pow(size);
DTYPE* d_arr = nullptr;

// Use unified memory to avoid explicit H2D copy timing
cudaMallocManaged(&d_arr, (size_t)padded_size * sizeof(DTYPE));
memcpy(d_arr, arrCpu, (size_t)size * sizeof(DTYPE));
for (int i = size; i < padded_size; i++) { d_arr[i] = INT_MAX; }
cudaMemPrefetchAsync(d_arr, padded_size * sizeof(DTYPE), 0);

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Perform bitonic sort on GPU
// Strategy: run global-memory phases while partners cross 4*blockDim tiles.
// Then run one batched shared-memory pass that completes remaining phases.
int thread_per_block = 1024;
int block_per_grid = (padded_size + thread_per_block - 1) / thread_per_block;
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
int min_num_block = prop.multiProcessorCount * 32;
if (block_per_grid < min_num_block) block_per_grid = min_num_block;

size_t sharedMem4x = (size_t)thread_per_block * 4 * sizeof(DTYPE);
cudaFuncSetCacheConfig(BitonicSort_shared_batched_4x, cudaFuncCachePreferShared);

for (int k = 2; k <= padded_size; k <<= 1) {
    int j = k >> 1;
    for (; j >= (thread_per_block << 2); j >>= 1) {
        BitonicSort_global<<<block_per_grid, thread_per_block>>>(d_arr, j, k, padded_size);
    }
    if (j > 0) {
        int blocks4x = (padded_size + (thread_per_block << 2) - 1) / (thread_per_block << 2);
        if (blocks4x < prop.multiProcessorCount * 8) blocks4x = prop.multiProcessorCount * 8;
        BitonicSort_shared_batched_4x<<<blocks4x, thread_per_block, sharedMem4x>>>(d_arr, k, padded_size);
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
