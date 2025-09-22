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

// GPU Kernel for bitonic sort with global memory
__global__ void BitonicSort_global(int* data, int j, int k, int size){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i >= size) return;
  
  int partnerGlobalIdx = i ^ j;
  
  if (i < partnerGlobalIdx && i < size && partnerGlobalIdx < size) {
    bool ascending = (i & k) == 0; 
    
    // Coalesced memory access pattern
    int val1 = data[i];
    int val2 = data[partnerGlobalIdx];

    // Compare and swap if needed
    if ((val1 > val2) == ascending) {
      data[i] = val2;
      data[partnerGlobalIdx] = val1;
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
    
    // Compare and swap if needed
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
  
  // Write back to global memory
  if (i < size) {
    data[i] = s_array[threadIdx.x];
  }
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

    // Perform bitonic sort on GPU using shared memory
    LOG_STAGE("Starting shared memory bitonic sort on GPU");
    // Optimal launch configuration for H100
    int threadsPerBlock = 512;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    LOG_INFO("Launch configuration: %d blocks, %d threads per block", blocksPerGrid, threadsPerBlock);

    // Calculate shared memory size needed
    size_t sharedMemSize = threadsPerBlock * sizeof(DTYPE);
    LOG_DEBUG("Shared memory size: %zu bytes", sharedMemSize);

    // Use smart hybrid approach: shared memory for small arrays, global memory for large arrays
    int step_count = 0;
    if (size <= 512) {
        // Small arrays: Use shared memory kernel (benefits from shared memory)
        LOG_DEBUG("Using shared memory kernel for small array");
        for (int k = 2; k <= size; k *= 2) {
            for (int j = k/2; j > 0; j /= 2) {
                BitonicSort_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_arr, j, k, size);
                step_count++;
                LOG_DEBUG("Step %d: Using shared memory kernel (j=%d, k=%d)", step_count, j, k);
            }
        }
        LOG_INFO("Shared memory bitonic sort completed in %d steps", step_count);
    } else {
        // Large arrays: Use global memory kernel
        LOG_DEBUG("Using global memory kernel for large array");
        for (int k = 2; k <= size; k *= 2) {
            for (int j = k/2; j > 0; j /= 2) {
                BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
                step_count++;
            }
        }
        LOG_INFO("Global memory bitonic sort completed in %d steps", step_count);
    }

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
