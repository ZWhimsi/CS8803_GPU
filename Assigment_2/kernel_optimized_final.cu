#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <string.h>

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

// OPTIMIZATION 1: Unrolled global kernel for better instruction throughput
__global__ void BitonicSort_global(int* data, int j, int k, int size){
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  // Process multiple elements per thread
  #pragma unroll 2
  for (int i = tid; i < size; i += stride) {
    int partnerIdx = i ^ j;
    
    if (i < partnerIdx && partnerIdx < size) {
      bool ascending = (i & k) == 0; 
      
      // Coalesced memory access
      int val1 = data[i];
      int val2 = data[partnerIdx];

      // Compare and swap
      if ((val1 > val2) == ascending) {
        data[i] = val2;
        data[partnerIdx] = val1;
      }
    }
  }
}

// OPTIMIZATION 2: Shared memory kernel with better synchronization
__global__ void BitonicSort_shared(int* data, int j, int k, int size){
  extern __shared__ DTYPE s_array[];
  
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  // Load data into shared memory with padding
  if (i < size) {
    s_array[threadIdx.x] = data[i];
  } else {
    s_array[threadIdx.x] = INT_MAX;
  }
  __syncthreads();
  
  int partnerGlobalIdx = i ^ j;
  int partnerBlockIdx = partnerGlobalIdx / blockDim.x;
  int partnerLocalIdx = partnerGlobalIdx % blockDim.x;
  bool sameBlock = (partnerBlockIdx == blockIdx.x);
  
  if (i < partnerGlobalIdx && i < size && partnerGlobalIdx < size) {
    bool ascending = (i & k) == 0; 
    
    int val1, val2;
    if (sameBlock) {
      val1 = s_array[threadIdx.x];
      val2 = s_array[partnerLocalIdx];
    } else {
      val1 = s_array[threadIdx.x];
      val2 = data[partnerGlobalIdx];
    }
    
    if ((val1 > val2) == ascending) {
      if (sameBlock) {
        s_array[threadIdx.x] = val2;
        s_array[partnerLocalIdx] = val1;
      } else {
        s_array[threadIdx.x] = val2;
        data[partnerGlobalIdx] = val1;
      }
    }
  }
  
  __syncthreads();
  
  if (i < size) {
    data[i] = s_array[threadIdx.x];
  }
}

// Helper functions
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

bool isPowerOfTwo(int n) {
    return n && !(n & (n - 1));
}

int nextPowerOfTwo(int n) {
    if (isPowerOfTwo(n)) return n;
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// CPU bitonic sort for validation
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
    
    // For bitonic sort, size must be a power of 2
    int size = nextPowerOfTwo(original_size);
    bool needs_padding = (size != original_size);
    
    if (needs_padding) {
        LOG_INFO("Padding array from %d to %d elements (next power of 2)", original_size, size);
    }
    
    // Allocate host memory
    int* arrCpu = (int*)malloc(size * sizeof(int));
    if (!arrCpu) {
        LOG_ERROR("Failed to allocate memory for CPU array");
        return 1;
    }
    
    // Initialize array
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
    
    // OPTIMIZATION 3: Use pinned memory for both input and output
    int* h_arr_pinned = nullptr;
    int* h_result_pinned = nullptr;
    cudaStream_t stream = 0;
    
    cudaError_t pinned_in = cudaMallocHost(&h_arr_pinned, size * sizeof(int));
    cudaError_t pinned_out = cudaMallocHost(&h_result_pinned, size * sizeof(int));
    
    bool use_pinned = (pinned_in == cudaSuccess && pinned_out == cudaSuccess);
    
    if (use_pinned) {
        memcpy(h_arr_pinned, arrCpu, size * sizeof(int));
        CUDA_CHECK(cudaStreamCreate(&stream));
        LOG_DEBUG("Using pinned memory for both input and output");
    } else {
        // Fallback to regular memory
        h_arr_pinned = arrCpu;
        h_result_pinned = (int*)malloc(size * sizeof(int));
        LOG_DEBUG("Using regular memory");
    }

    LOG_STAGE("Setting up GPU memory and data transfer");
    
    DTYPE* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, size * sizeof(DTYPE)));

    // Initial copy
    LOG_DEBUG("Copying data from host to device");
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr_pinned, size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    LOG_INFO("Data successfully copied to GPU");

    /* ==== TIMING START ==== */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float h2dTime, kernelTime, d2hTime;

    // H2D Transfer
    cudaEventRecord(start);
    if (use_pinned && stream) {
        CUDA_CHECK(cudaMemcpyAsync(d_arr, h_arr_pinned, size * sizeof(DTYPE), 
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(d_arr, h_arr_pinned, size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    // Kernel execution
    cudaEventRecord(start);

    LOG_STAGE("Starting optimized bitonic sort on GPU");
    
    // OPTIMIZATION 4: Use 1024 threads for maximum occupancy on H100
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Cap grid size to reduce overhead
    if (blocksPerGrid > 32768) {
        blocksPerGrid = 32768;
    }
    
    LOG_INFO("Launch configuration: %d blocks, %d threads per block", blocksPerGrid, threadsPerBlock);
    
    size_t sharedMemSize = threadsPerBlock * sizeof(DTYPE);
    
    // OPTIMIZATION 5: Smart kernel selection based on problem size
    int step_count = 0;
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            step_count++;
            
            // Use shared memory only when we know partners are in same block
            if (j < threadsPerBlock && k <= threadsPerBlock * 2) {
                BitonicSort_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_arr, j, k, size);
            } else {
                BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
            }
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_INFO("Bitonic sort completed in %d steps", step_count);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // D2H Transfer
    cudaEventRecord(start);
    
    LOG_STAGE("Transferring sorted data back to host");
    if (use_pinned && stream) {
        CUDA_CHECK(cudaMemcpyAsync(h_result_pinned, d_arr, size * sizeof(DTYPE), 
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaMemcpy(h_result_pinned, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);
    /* ==== TIMING END ==== */

    LOG_INFO("Sorted array:");
    printArray(h_result_pinned, original_size, "  ");

    // CPU sort for validation
    LOG_STAGE("Performing CPU sort for comparison");
    int* arrCpuSorted = (int*)malloc(size * sizeof(int));
    memcpy(arrCpuSorted, arrCpu, size * sizeof(int));
    
    clock_t cpuStart = clock();
    bitonicSortCPU(arrCpuSorted, size);
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;

    // Validate results (only original elements)
    LOG_STAGE("Validating results");
    bool valid = true;
    for (int i = 0; i < original_size; i++) {
        if (h_result_pinned[i] != arrCpuSorted[i]) {
            LOG_ERROR("Mismatch at index %d: GPU=%d, CPU=%d", i, h_result_pinned[i], arrCpuSorted[i]);
            valid = false;
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
        
        if (elementsPerSecond > 900) {
            printf("PERF PASSING\n");
        } else {
            printf("PERF FAILING (need > 900 MOPE/s, got %.2f)\n", elementsPerSecond);
        }
        
        printf("GPU Sort is %3.0fx faster than CPU !!!\n", cpuTime / gpuTime);
        printf("H2D Transfer Time (ms): %f\n", h2dTime);
        printf("Kernel Time (ms)      : %f\n", kernelTime);
        printf("D2H Transfer Time (ms): %f\n", d2hTime);
        
        printf("\nOptimization Summary:\n");
        printf("- Array padding: %s (from %d to %d)\n", needs_padding ? "YES" : "NO", original_size, size);
        printf("- Pinned memory: %s\n", use_pinned ? "YES (both input/output)" : "NO");
        printf("- Thread configuration: %d threads/block (was 512)\n", threadsPerBlock);
        printf("- Kernel launches: %d\n", step_count);
        printf("- Smart kernel selection: YES\n");
        printf("- Loop unrolling: YES (2x)\n");
    } else {
        LOG_ERROR("FUNCTIONAL FAIL");
        printf("FUNCTIONAL FAIL\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    if (stream) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    if (use_pinned) {
        CUDA_CHECK(cudaFreeHost(h_arr_pinned));
        CUDA_CHECK(cudaFreeHost(h_result_pinned));
    } else {
        free(h_result_pinned);
    }
    
    free(arrCpu);
    free(arrCpuSorted);
    
    return valid ? 0 : 1;
}
