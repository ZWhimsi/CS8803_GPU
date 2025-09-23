#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef int DTYPE;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, error, cudaGetErrorString(error), #call); \
            exit(1); \
        } \
    } while (0)

// Logging macros for better debugging
#define LOG_INFO(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_STAGE(stage) printf("\n[STAGE] %s\n", stage)

// Diagnostics toggle (enable with -DENABLE_DIAG=1)
#ifndef ENABLE_DIAG
#define ENABLE_DIAG 0
#endif

// Limit sampling to first N blocks to reduce overhead
#ifndef DIAG_SAMPLE_BLOCKS
#define DIAG_SAMPLE_BLOCKS 128
#endif

typedef struct {
  unsigned long long compares;
  unsigned long long swaps;
  unsigned long long sameBlockPairs;
  unsigned long long crossBlockPairs;
  unsigned long long divergentWarps;
} KernelStats;

#if ENABLE_DIAG
__device__ KernelStats d_stats;
#endif

// GPU Kernel for bitonic sort with global memory
__launch_bounds__(1024, 2)
__global__ void BitonicSort_global(int* __restrict__ data, int j, int k, int size){
  const int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  #pragma unroll 2
  for (int i = threadId; i < size; i += stride) {
    const int partnerGlobalIdx = i ^ j;

    if (i < partnerGlobalIdx && partnerGlobalIdx < size) {
      const bool ascending = (i & k) == 0;

      int val1 = data[i];
      int val2 = data[partnerGlobalIdx];

      #if ENABLE_DIAG
      if (blockIdx.x < DIAG_SAMPLE_BLOCKS) {
        const int partnerBlockIdx = partnerGlobalIdx / blockDim.x;
        const bool sameBlock = (partnerBlockIdx == blockIdx.x);
        atomicAdd(&d_stats.compares, 1ULL);
        if (sameBlock) {
          atomicAdd(&d_stats.sameBlockPairs, 1ULL);
        } else {
          atomicAdd(&d_stats.crossBlockPairs, 1ULL);
        }
        unsigned int mask = __ballot_sync(0xffffffff, (val1 > val2) == ascending);
        if ((threadIdx.x & 31) == 0) {
          if (mask != 0u && mask != 0xffffffffu) {
            atomicAdd(&d_stats.divergentWarps, 1ULL);
          }
        }
      }
      #endif

      if ((val1 > val2) == ascending) {
        data[i] = val2;
        data[partnerGlobalIdx] = val1;
        #if ENABLE_DIAG
        if (blockIdx.x < DIAG_SAMPLE_BLOCKS) {
          atomicAdd(&d_stats.swaps, 1ULL);
        }
        #endif
      }
    }
  }
}

// GPU Kernel for bitonic sort with shared memory
__global__ void BitonicSort_shared(int* data, int j, int k, int size){
  extern __shared__ DTYPE s_array[];
  
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  // Handle threads beyond array size
  if (i >= size) return;
  
  // Load data into shared memory
  s_array[threadIdx.x] = data[i];
  __syncthreads();
  
  int partnerGlobalIdx = i ^ j;
  int partnerBlockIdx = partnerGlobalIdx / blockDim.x;
  int partnerLocalIdx = partnerGlobalIdx % blockDim.x;
  bool sameBlock = (partnerBlockIdx == blockIdx.x);
  
  if (i < partnerGlobalIdx && i < size && partnerGlobalIdx < size) {
    bool ascending = (i & k) == 0; 
    
    int val1, val2;
    if (sameBlock) {
      // Both elements are in the same block
      val1 = s_array[threadIdx.x];
      val2 = s_array[partnerLocalIdx];
    } else {
      // Partner is in a different block, access global memory
      val1 = s_array[threadIdx.x];
      val2 = data[partnerGlobalIdx];
    }
    
    #if ENABLE_DIAG
    if (blockIdx.x < DIAG_SAMPLE_BLOCKS) {
      atomicAdd(&d_stats.compares, 1ULL);
      if (sameBlock) {
        atomicAdd(&d_stats.sameBlockPairs, 1ULL);
      } else {
        atomicAdd(&d_stats.crossBlockPairs, 1ULL);
      }
      unsigned int mask = __ballot_sync(0xffffffff, (val1 > val2) == ascending);
      if ((threadIdx.x & 31) == 0) {
        if (mask != 0u && mask != 0xffffffffu) {
          atomicAdd(&d_stats.divergentWarps, 1ULL);
        }
      }
    }
    #endif

    // Compare and swap if needed
    if ((val1 > val2) == ascending) {
      if (sameBlock) {
        s_array[threadIdx.x] = val2;
        s_array[partnerLocalIdx] = val1;
      } else {
        s_array[threadIdx.x] = val2;
        data[partnerGlobalIdx] = val1;
      }
      #if ENABLE_DIAG
      if (blockIdx.x < DIAG_SAMPLE_BLOCKS) {
        atomicAdd(&d_stats.swaps, 1ULL);
      }
      #endif
    }
  }
  
  __syncthreads();
  
  // Write back to global memory
  if (i < size) {
    data[i] = s_array[threadIdx.x];
  }
}

// GPU Kernel that batches all intra-block (j < blockDim.x) phases for a given k
__global__ void BitonicSort_shared_batched(int* __restrict__ data, int k, int size){
  extern __shared__ DTYPE s_array[];

  const int globalBase = blockIdx.x * blockDim.x;
  const int i = globalBase + threadIdx.x;
  const bool ascending_const = ((i & k) == 0);

  // Load into shared, pad out-of-range with INT_MAX
  if (i < size) {
    s_array[threadIdx.x] = data[i];
  } else {
    s_array[threadIdx.x] = INT_MAX;
  }
  __syncthreads();

  // Process all remaining j within the block for this k
  #pragma unroll
  for (int jj = min(k >> 1, blockDim.x >> 1); jj > 0; jj >>= 1) {
    const int partnerLocalIdx = threadIdx.x ^ jj;

    // Only one of each pair performs the compare-swap
    if (threadIdx.x < partnerLocalIdx) {
      int val1 = s_array[threadIdx.x];
      int val2 = s_array[partnerLocalIdx];

      if ((val1 > val2) == ascending_const) {
        s_array[threadIdx.x] = val2;
        s_array[partnerLocalIdx] = val1;
      }
    }
    __syncthreads();
  }

  // Write back
  if (i < size) {
    data[i] = s_array[threadIdx.x];
  }
}

// GPU Kernel that batches j phases for a 2x tile (handles j < 2*blockDim.x)
__global__ void BitonicSort_shared_batched_2x(int* __restrict__ data, int k, int size){
  extern __shared__ DTYPE s_array[]; // size: 2 * blockDim.x

  const int base = (blockIdx.x * blockDim.x) << 1; // 2 * blockDim per block
  const int tid  = threadIdx.x;

  // Two logical indices per thread
  const int g0 = base + tid;
  const int g1 = g0 + blockDim.x;

  // Load two elements
  s_array[tid] = (g0 < size) ? data[g0] : INT_MAX;
  s_array[tid + blockDim.x] = (g1 < size) ? data[g1] : INT_MAX;
  __syncthreads();

  // Process all jj within 2*blockDim for this k
  for (int jj = min(k >> 1, blockDim.x); jj > 0; jj >>= 1) {
    // First logical index
    {
      const int lid = tid;
      const int partner = lid ^ jj;
      if (lid < partner) {
        const int gi = base + lid;
        const bool ascending = ((gi & k) == 0);
        int a = s_array[lid];
        int b = s_array[partner];
        if ((a > b) == ascending) {
          s_array[lid] = b;
          s_array[partner] = a;
        }
      }
    }
    __syncthreads();

    // Second logical index
    {
      const int lid = tid + blockDim.x;
      const int partner = lid ^ jj;
      if (lid < partner) {
        const int gi = base + lid;
        const bool ascending = ((gi & k) == 0);
        int a = s_array[lid];
        int b = s_array[partner];
        if ((a > b) == ascending) {
          s_array[lid] = b;
          s_array[partner] = a;
        }
      }
    }
    __syncthreads();
  }

  // Store back
  if (g0 < size) data[g0] = s_array[tid];
  if (g1 < size) data[g1] = s_array[tid + blockDim.x];
}

// GPU Kernel that batches j phases for a 4x tile (handles j < 4*blockDim.x)
__global__ void BitonicSort_shared_batched_4x(int* __restrict__ data, int k, int size){
  extern __shared__ DTYPE s_array[]; // size: 4 * blockDim.x

  const int bd = blockDim.x;
  const int base = (blockIdx.x * bd) << 2; // 4 * blockDim per block
  const int tid  = threadIdx.x;

  const int g0 = base + tid;
  const int g1 = g0 + bd;
  const int g2 = g1 + bd;
  const int g3 = g2 + bd;

  // Load four elements into shared memory
  s_array[tid]          = (g0 < size) ? data[g0] : INT_MAX;
  s_array[tid + bd]     = (g1 < size) ? data[g1] : INT_MAX;
  s_array[tid + 2 * bd] = (g2 < size) ? data[g2] : INT_MAX;
  s_array[tid + 3 * bd] = (g3 < size) ? data[g3] : INT_MAX;
  __syncthreads();

  for (int jj = min(k >> 1, 2 * bd); jj > 0; jj >>= 1) {
    // Process all four logical indices in the 4*bd tile
    #define PROCESS_LID(LID_EXPR) \
      { \
        const int lid = (LID_EXPR); \
        const int partner = lid ^ jj; \
        if (lid < partner) { \
          const int gi = base + lid; \
          const bool ascending = ((gi & k) == 0); \
          int a = s_array[lid]; \
          int b = s_array[partner]; \
          if ((a > b) == ascending) { \
            s_array[lid] = b; \
            s_array[partner] = a; \
          } \
        } \
      }

    PROCESS_LID(tid);
    __syncthreads();
    PROCESS_LID(tid + bd);
    __syncthreads();
    PROCESS_LID(tid + 2 * bd);
    __syncthreads();
    PROCESS_LID(tid + 3 * bd);
    __syncthreads();
    #undef PROCESS_LID
  }

  // Store back
  if (g0 < size) data[g0] = s_array[tid];
  if (g1 < size) data[g1] = s_array[tid + bd];
  if (g2 < size) data[g2] = s_array[tid + 2 * bd];
  if (g3 < size) data[g3] = s_array[tid + 3 * bd];
}

// Helper function to print array
void printArray(int* arr, int size, const char* prefix = "") {
    printf("%s[", prefix);
    int print_limit = (size > 20) ? 20 : size;
    for (int i = 0; i < print_limit; i++) {
        printf("%d", arr[i]);
        if (i < print_limit - 1) printf(", ");
    }
    if (size > 20) {
        printf("... (%d more elements)", size - 20);
    }
    printf("]\n");
}

// Check if a number is a power of 2
bool isPowerOfTwo(int n) {
    return n && !(n & (n - 1));
}

// Find the next power of 2 greater than or equal to n
int nextPowerOfTwo(int n) {
    if (isPowerOfTwo(n)) return n;
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// CPU implementation of bitonic sort for validation
void bitonicSortCPU(int* arr, int size) {
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for (int i = 0; i < size; i++) {
                int partnerIdx = i ^ j;
                if (i < partnerIdx && partnerIdx < size) {
                    bool ascending = (i & k) == 0;
                    if ((arr[i] > arr[partnerIdx]) == ascending) {
                        int temp = arr[i];
                        arr[i] = arr[partnerIdx];
                        arr[partnerIdx] = temp;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        LOG_ERROR("Usage: %s <array_size>", argv[0]);
        return 1;
    }
    
    int original_size = atoi(argv[1]);
    if (original_size <= 0) {
        LOG_ERROR("Array size must be positive");
        return 1;
    }
    
    LOG_INFO("Starting bitonic sort with array size: %d", original_size);
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    LOG_INFO("Device: %s | SMs=%d | MaxThreads/Block=%d | MaxThreads/SM=%d | SharedMem/Block=%zu KB",
             prop.name, prop.multiProcessorCount, prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor,
             prop.sharedMemPerBlock / 1024);
    
    // For bitonic sort, size must be a power of 2
    int size = nextPowerOfTwo(original_size);
    bool needs_padding = (size != original_size);
    
    if (needs_padding) {
        LOG_INFO("Padding array from %d to %d elements (next power of 2)", original_size, size);
    }
    
    // Allocate host memory for the array
    int* arrCpu = (int*)malloc(size * sizeof(int));
    if (!arrCpu) {
        LOG_ERROR("Failed to allocate memory for CPU array");
        return 1;
    }
    
    // Initialize array with random values
    LOG_STAGE("Generating random array");
    srand(time(NULL));
    for (int i = 0; i < original_size; i++) {
        arrCpu[i] = rand() % 1000;
    }
    
    // Pad with INT_MAX if needed
    for (int i = original_size; i < size; i++) {
        arrCpu[i] = INT_MAX;
    }
    
    LOG_INFO("Original array:");
    printArray(arrCpu, original_size, "  ");
    
    // Allocate memory for GPU result using pinned memory for faster D2H
    int* arrSortedGpu = nullptr;
    cudaError_t pinned_result = cudaMallocHost(&arrSortedGpu, size * sizeof(int));
    bool use_pinned_output = (pinned_result == cudaSuccess);
    
    if (!use_pinned_output) {
        LOG_DEBUG("Pinned memory for output failed, using regular memory");
        arrSortedGpu = (int*)malloc(size * sizeof(int));
        if (!arrSortedGpu) {
            LOG_ERROR("Failed to allocate memory for GPU result array");
            free(arrCpu);
            return 1;
        }
    } else {
        LOG_DEBUG("Using pinned memory for output (faster D2H transfers)");
    }

    LOG_STAGE("Setting up GPU memory and data transfer");

    // Transfer data (arr_cpu) to device
    DTYPE* d_arr;
    
    // Try pinned memory for input
    int* h_arr_pinned = nullptr;
    cudaError_t input_pinned = cudaMallocHost(&h_arr_pinned, size * sizeof(int));
    bool use_pinned_input = (input_pinned == cudaSuccess);
    
    if (use_pinned_input) {
        memcpy(h_arr_pinned, arrCpu, size * sizeof(int));
        LOG_DEBUG("Using pinned memory for input");
    } else {
        LOG_DEBUG("Using regular memory for input");
    }
    
    CUDA_CHECK(cudaMalloc(&d_arr, size * sizeof(DTYPE)));

    // Copy data from host to device
    LOG_DEBUG("Copying data from host to device");
    CUDA_CHECK(cudaMemcpy(d_arr, use_pinned_input ? h_arr_pinned : arrCpu, 
                          size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    LOG_INFO("Data successfully copied to GPU");

    #if ENABLE_DIAG
    KernelStats zero_stats = {};
    CUDA_CHECK(cudaMemcpyToSymbol(d_stats, &zero_stats, sizeof(KernelStats)));
    #endif

    /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float h2dTime, kernelTime, d2hTime;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_arr, use_pinned_input ? h_arr_pinned : arrCpu, 
                          size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

  // Perform bitonic sort on GPU (hybrid: global for j>=blockDim, batched shared for remaining j)
  LOG_STAGE("Starting hybrid bitonic sort on GPU");

  // H100: favor 1024 threads per block; ensure enough blocks to saturate SMs
  int threadsPerBlock = 1024;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  int minBlocks = prop.multiProcessorCount * 32; // aim for higher residency on H100
  if (blocksPerGrid < minBlocks) blocksPerGrid = minBlocks;
  LOG_INFO("Launch configuration: %d blocks, %d threads per block", blocksPerGrid, threadsPerBlock);

  size_t sharedMemSize   = threadsPerBlock * sizeof(DTYPE);
  size_t sharedMemSize2x = 2 * threadsPerBlock * sizeof(DTYPE);
  size_t sharedMemSize4x = 4 * threadsPerBlock * sizeof(DTYPE);
  LOG_DEBUG("Shared memory size: %zu | 2x: %zu | 4x: %zu bytes", sharedMemSize, sharedMemSize2x, sharedMemSize4x);

  int step_count = 0;
  for (int k = 2; k <= size; k <<= 1) {
    int j = k >> 1;

    // Global passes while partners cross 4x tiles
    for (; j >= (threadsPerBlock << 2); j >>= 1) {
      BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
      step_count++;
    }

    // One batched shared-memory 4x-tile pass per k for remaining j
    if (j > 0) {
      // Compute blocks from problem size and tile size to cover domain
      int blocks4x = (size + (threadsPerBlock << 2) - 1) / (threadsPerBlock << 2);
      if (blocks4x < prop.multiProcessorCount * 8) blocks4x = prop.multiProcessorCount * 8; // keep SMs busy
      // Prefer shared memory for batched kernels
      cudaFuncSetCacheConfig(BitonicSort_shared_batched_4x, cudaFuncCachePreferShared);
      BitonicSort_shared_batched_4x<<<blocks4x, threadsPerBlock, sharedMemSize4x>>>(d_arr, k, size);
      step_count++;
    }
  }
  LOG_INFO("Hybrid bitonic sort completed in %d steps", step_count);

    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_DEBUG("Shared memory kernel completed");
    LOG_INFO("Shared memory bitonic sort completed");

    /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaEventRecord(start);
    /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    // Transfer sorted data back to host (copied to arrSortedGpu)
    LOG_STAGE("Transferring sorted data back to host");
    
    // OPTIMIZATION: Use async copy if we have pinned memory
    if (use_pinned_output) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMemcpyAsync(arrSortedGpu, d_arr, size * sizeof(DTYPE), 
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    
    LOG_INFO("Sorted array:");
    printArray(arrSortedGpu, original_size, "  ");

    /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);
    /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    // Derived throughput metrics
    double totalBytes = (double)size * sizeof(DTYPE);
    double h2dGBs = totalBytes / (1e6 * h2dTime);
    double d2hGBs = totalBytes / (1e6 * d2hTime);

    // Occupancy estimation
    int numBlocksPerSm = 0;
    size_t occupancySharedMem = 512 * sizeof(DTYPE); // matches threadsPerBlock when size<=512
    if (size <= 512) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, BitonicSort_shared, 512, (int)occupancySharedMem));
    } else {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, BitonicSort_global, 512, 0));
    }
    double occPct = 100.0 * (double)numBlocksPerSm * 512.0 / (double)prop.maxThreadsPerMultiProcessor;

    printf("\nPerf Diagnostics (host):\n");
    printf("- Kernel launches (steps): %d\n", step_count);
    printf("- Estimated occupancy: %.1f%% (blocks/SM=%d)\n", occPct, numBlocksPerSm);
    printf("- H2D throughput: %.2f GB/s | D2H throughput: %.2f GB/s\n", h2dGBs, d2hGBs);

    #if ENABLE_DIAG
    // Read device-side stats and scale by sampling ratio
    KernelStats h_stats;
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_stats, d_stats, sizeof(KernelStats)));
    int sampledBlocks = (blocksPerGrid < DIAG_SAMPLE_BLOCKS) ? blocksPerGrid : DIAG_SAMPLE_BLOCKS;
    double scale = sampledBlocks > 0 ? ((double)blocksPerGrid / (double)sampledBlocks) : 1.0;
    unsigned long long est_compares = (unsigned long long)(h_stats.compares * scale + 0.5);
    unsigned long long est_swaps = (unsigned long long)(h_stats.swaps * scale + 0.5);
    unsigned long long est_same = (unsigned long long)(h_stats.sameBlockPairs * scale + 0.5);
    unsigned long long est_cross = (unsigned long long)(h_stats.crossBlockPairs * scale + 0.5);
    unsigned long long divWarps = (unsigned long long)(h_stats.divergentWarps * scale + 0.5);

    printf("\nPerf Diagnostics (device-sampled):\n");
    printf("- Compares (estimated): %llu\n", est_compares);
    printf("- Swaps (estimated):    %llu\n", est_swaps);
    printf("- Same-block pairs est: %llu | Cross-block pairs est: %llu\n", est_same, est_cross);
    printf("- Divergent warps est:  %llu\n", divWarps);
    #endif
    /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    // CPU sort for validation
    LOG_STAGE("Performing CPU sort for comparison");
    int* arrCpuSorted = (int*)malloc(size * sizeof(int));
    memcpy(arrCpuSorted, arrCpu, size * sizeof(int));
    
    clock_t cpuStart = clock();
    bitonicSortCPU(arrCpuSorted, size);
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;

    // Validate results (only compare original_size elements)
    LOG_STAGE("Validating results");
    bool valid = true;
    for (int i = 0; i < original_size; i++) {
        if (arrSortedGpu[i] != arrCpuSorted[i]) {
            LOG_ERROR("Mismatch at index %d: GPU=%d, CPU=%d", i, arrSortedGpu[i], arrCpuSorted[i]);
            valid = false;
            
            // Show context around mismatch
            int start = (i > 5) ? i - 5 : 0;
            int end = (i + 5 < original_size) ? i + 5 : original_size;
            
            printf("[DEBUG] GPU array around mismatch: [");
            for (int j = start; j < end; j++) {
                printf("%d", arrSortedGpu[j]);
                if (j < end - 1) printf(", ");
            }
            printf("]\n");
            
            printf("[DEBUG] CPU array around mismatch: [");
            for (int j = start; j < end; j++) {
                printf("%d", arrCpuSorted[j]);
                if (j < end - 1) printf(", ");
            }
            printf("]\n");
            
            LOG_ERROR("Validation failed - results don't match at index %d", i);
            break;
        }
    }
    
    if (valid) {
        LOG_INFO("Validation successful - GPU and CPU results match!");
        LOG_INFO("FUNCTIONAL SUCCESS");
        printf("FUNCTIONAL SUCCESS\n");
        
        double gpuTime = h2dTime + kernelTime + d2hTime;
        double elementsPerSecond = (original_size / (gpuTime / 1000.0)) / 1e6;
        
        printf("Array size         : %d\n", original_size);
        printf("CPU Sort Time (ms) : %f\n", cpuTime);
        printf("GPU Sort Time (ms) : %f\n", gpuTime);
        printf("GPU Sort Speed     : %f million elements per second\n", elementsPerSecond);
        
        if (elementsPerSecond > 1000) {
            printf("PERF PASSING\n");
        } else {
            printf("PERF FAILING (need > 1000 MOPE/s, got %.2f)\n", elementsPerSecond);
        }
        
        printf("GPU Sort is %3.0fx faster than CPU !!!\n", cpuTime / gpuTime);
        printf("H2D Transfer Time (ms): %f\n", h2dTime);
        printf("Kernel Time (ms)      : %f\n", kernelTime);
        printf("D2H Transfer Time (ms): %f\n", d2hTime);
        
        // Debug info
        printf("\nOptimization Status:\n");
        printf("- Input pinned memory: %s\n", use_pinned_input ? "YES" : "NO");
        printf("- Output pinned memory: %s\n", use_pinned_output ? "YES" : "NO");
    } else {
        LOG_ERROR("FUNCTIONAL FAIL");
    }

    // Clean up GPU memory
    CUDA_CHECK(cudaFree(d_arr));
    if (use_pinned_input) {
        CUDA_CHECK(cudaFreeHost(h_arr_pinned));
    }
    if (use_pinned_output) {
        CUDA_CHECK(cudaFreeHost(arrSortedGpu));
    } else {
        free(arrSortedGpu);
    }
    
    free(arrCpu);
    free(arrCpuSorted);
    
    return valid ? 0 : 1;
}
