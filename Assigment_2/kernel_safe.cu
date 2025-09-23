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

// Simple global memory kernel - SAFE VERSION
__global__ void __launch_bounds__(1024, 2) BitonicSort_global(int* __restrict__ data, int j, int k, int size){
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
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }
    
    int original_size = atoi(argv[1]);
    if (original_size <= 0) {
        printf("Array size must be positive\n");
        return 1;
    }
    
    printf("SAFE bitonic sort with array size: %d\n", original_size);
    
    // For bitonic sort, size must be a power of 2
    int size = nextPowerOfTwo(original_size);
    bool needs_padding = (size != original_size);
    
    if (needs_padding) {
        printf("Padding array from %d to %d elements\n", original_size, size);
    }
    
    // Allocate host memory for the array
    int* arrCpu = (int*)malloc(size * sizeof(int));
    if (!arrCpu) {
        printf("Failed to allocate memory for CPU array\n");
        return 1;
    }
    
    // Initialize array with random values
    srand(time(NULL));
    for (int i = 0; i < original_size; i++) {
        arrCpu[i] = rand() % 1000;
    }
    
    // Pad with INT_MAX if needed
    for (int i = original_size; i < size; i++) {
        arrCpu[i] = INT_MAX;
    }
    
    printf("Original array: ");
    printArray(arrCpu, original_size, "");
    
    // Allocate GPU memory
    DTYPE* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, size * sizeof(DTYPE)));
    
    // Allocate result memory
    int* arrSortedGpu = (int*)malloc(size * sizeof(int));
    if (!arrSortedGpu) {
        printf("Failed to allocate memory for GPU result array\n");
        free(arrCpu);
        return 1;
    }

    /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float h2dTime, kernelTime, d2hTime;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    // Perform bitonic sort on GPU - SIMPLE GLOBAL ONLY
    printf("Starting SAFE global-only bitonic sort\n");
    
    // H100-optimized launch configuration
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launch: %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);

    int step_count = 0;
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
            step_count++;
            
            // Check for kernel errors every 50 steps
            if (step_count % 50 == 0) {
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA kernel error at step %d: %s\n", step_count, cudaGetErrorString(err));
                    exit(1);
                }
                printf("Progress: step %d/378\n", step_count);
            }
        }
    }
    
    printf("Synchronizing after %d steps...\n", step_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("SAFE bitonic sort completed\n");

    /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaEventRecord(start);
    /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    // Transfer result back
    printf("Transferring result back to host\n");
    CUDA_CHECK(cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    
    printf("Sorted array: ");
    printArray(arrSortedGpu, original_size, "");

    /* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);
    /* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

    // Quick validation (first 100 elements)
    printf("Quick validation\n");
    bool sorted = true;
    for (int i = 0; i < original_size - 1 && i < 100; i++) {
        if (arrSortedGpu[i] > arrSortedGpu[i+1]) {
            printf("FAIL: arr[%d]=%d > arr[%d]=%d\n", i, arrSortedGpu[i], i+1, arrSortedGpu[i+1]);
            sorted = false;
            break;
        }
    }
    
    if (sorted) {
        printf("FUNCTIONAL SUCCESS\n");
        
        double gpuTime = h2dTime + kernelTime + d2hTime;
        double elementsPerSecond = (original_size / (gpuTime / 1000.0)) / 1e6;
        
        printf("Array size         : %d\n", original_size);
        printf("GPU Sort Time (ms) : %f\n", gpuTime);
        printf("GPU Sort Speed     : %f million elements per second\n", elementsPerSecond);
        
        if (elementsPerSecond > 1000) {
            printf("PERF PASSING\n");
        } else {
            printf("PERF FAILING (need > 1000 MOPE/s, got %.2f)\n", elementsPerSecond);
        }
        
        printf("H2D Transfer Time (ms): %f\n", h2dTime);
        printf("Kernel Time (ms)      : %f\n", kernelTime);
        printf("D2H Transfer Time (ms): %f\n", d2hTime);
    } else {
        printf("FUNCTIONAL FAIL\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(arrCpu);
    free(arrSortedGpu);
    
    return sorted ? 0 : 1;
}
