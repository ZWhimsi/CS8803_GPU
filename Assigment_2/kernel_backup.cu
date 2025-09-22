
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
// ==== DO NOT MODIFY CODE ABOVE THIS LINE ====

#define DTYPE int
// Add any additional #include headers or helper macros needed

// Logging and debugging macros
#define LOG_INFO(msg, ...) printf("[INFO] " msg "\n", ##__VA_ARGS__)
#define LOG_STAGE(msg, ...) printf("[STAGE] " msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) printf("[ERROR] " msg "\n", ##__VA_ARGS__)
#define LOG_DEBUG(msg, ...) printf("[DEBUG] " msg "\n", ##__VA_ARGS__)

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        LOG_ERROR("CUDA Error: %s at %s:%d", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Function to print array with optional prefix
void printArray(DTYPE* arr, int size, const char* prefix = "", int maxElements = 20) {
    if (prefix[0] != '\0') {
        printf("%s", prefix);
    }
    printf("[");
    int printSize = (size > maxElements) ? maxElements : size;
    for (int i = 0; i < printSize; i++) {
        printf("%d", arr[i]);
        if (i < printSize - 1) printf(", ");
    }
    if (size > maxElements) {
        printf("... (%d more elements)", size - maxElements);
    }
    printf("]\n");
}

// Function to print sub-array for debugging
void printSubArray(DTYPE* arr, int start, int end, const char* prefix = "") {
    if (prefix[0] != '\0') {
        printf("%s", prefix);
    }
    printf("[");
    for (int i = start; i < end && i < start + 20; i++) {
        printf("%d", arr[i]);
        if (i < end - 1) printf(", ");
    }
    if (end - start > 20) {
        printf("... (%d more elements)", end - start - 20);
    }
    printf("]\n");
}


// Implement your GPU device kernel(s) here (e.g., the bitonic sort kernel).
__global__ void bitonicSort_naive(int *data, int j, int k) {
    // Get the global index for this thread
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine the index of the element to compare with
    unsigned int ixj = i ^ j; // XOR to find the partner element

    // Only compare if the partner is after us in the array
    if (ixj > i) {
        // FIXED: Correct sort direction calculation for bitonic sort
        // The direction is determined by the bit pattern of i and k
        int d = (i & k) == 0; 
        
        // Load two values from slow global memory
        int val1 = data[i];
        int val2 = data[ixj];

        // Compare and swap if needed
        if ((val1 > val2) == d) {
            // Write two values back to slow global memory
            data[i] = val2;
            data[ixj] = val1;
        }
    }
}


// Enhanced kernel with logging for debugging
__global__ void bitonicSort_with_logging(int *data, int j, int k, int array_size, int step_count) {
    // Get the global index for this thread
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Log from first few threads to see if kernel is executing
    if (i < 4) {
        printf("[KERNEL] Thread %d: Step %d, j=%d, k=%d, array_size=%d\n", i, step_count, j, k, array_size);
    }

    // Determine the index of the element to compare with
    unsigned int ixj = i ^ j; // XOR to find the partner element

    // Debug: Show XOR calculations for first few threads
    if (i < 4) {
        printf("[KERNEL] Thread %d: i=%d, j=%d, ixj=%d, condition ixj>i: %s\n", 
               i, i, j, ixj, (ixj > i) ? "true" : "false");
    }

    // Only compare if the partner is after us in the array
    if (ixj > i && i < array_size && ixj < array_size) {
        // FIXED: Correct sort direction calculation for bitonic sort
        // The direction is determined by the bit pattern of i and k
        int d = (i & k) == 0; 
        
        // Load two values from slow global memory
        int val1 = data[i];
        int val2 = data[ixj];

        // Log comparison details for first few threads
        if (i < 4) {
            printf("[KERNEL] Thread %d: comparing data[%d]=%d with data[%d]=%d, direction=%d\n", 
                   i, i, val1, ixj, val2, d);
        }

        // Compare and swap if needed
        if ((val1 > val2) == d) {
            // Write two values back to slow global memory
            data[i] = val2;
            data[ixj] = val1;
            
            // Log swap for first few threads
            if (i < 4) {
                printf("[KERNEL] Thread %d: SWAPPED data[%d]=%d <-> data[%d]=%d\n", 
                       i, i, val2, ixj, val1);
            }
        }
    }
}




__global__ void BitonicSort_shared(int* data, int j, int k, int size){
  extern __shared__ DTYPE s_array[];
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  // Load data into shared memory for all threads in block
  if (i < size) {
    s_array[threadIdx.x] = data[i];
  } else {
    s_array[threadIdx.x] = INT_MAX;  // Pad with max value for threads beyond array
  }
  __syncthreads();

  int partnerGlobalIdx = i ^ j;
  int partnerBlockIdx = partnerGlobalIdx / blockDim.x;
  int partnerLocalIdx = partnerGlobalIdx % blockDim.x;
  
  // Check if partner is within the same block
  bool sameBlock = (partnerBlockIdx == blockIdx.x);
  
  if (i < partnerGlobalIdx && i < size && partnerGlobalIdx < size) {
    bool ascending = (i & k) == 0; 
    
    int val1, val2;
    
    if (sameBlock) {
      // Both threads in same block - use shared memory
      val1 = s_array[threadIdx.x];
      val2 = s_array[partnerLocalIdx];
    } else {
      // Partner in different block - use global memory
      val1 = s_array[threadIdx.x];
      val2 = data[partnerGlobalIdx];
    }

    // Compare and swap if needed
    if ((val1 > val2) == ascending) {
      if (sameBlock) {
        // Both in same block - update shared memory
        s_array[threadIdx.x] = val2;
        s_array[partnerLocalIdx] = val1;
      } else {
        // Partner in different block - update global memory
        s_array[threadIdx.x] = val2;
        data[partnerGlobalIdx] = val1;
      }
    }
  }
  __syncthreads();
  
  // Write result back to global memory
  if (i < size) {
    data[i] = s_array[threadIdx.x];
  }
}

// Pure global memory kernel for large arrays
__global__ void BitonicSort_global(int* data, int j, int k, int size){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  // Only process threads that correspond to valid array elements
  if (i >= size) return;
  
  int partnerGlobalIdx = i ^ j;
  
  if (i < partnerGlobalIdx && i < size && partnerGlobalIdx < size) {
    bool ascending = (i & k) == 0; 
    
    int val1 = data[i];
    int val2 = data[partnerGlobalIdx];

    // Compare and swap if needed
    if ((val1 > val2) == ascending) {
      data[i] = val2;
      data[partnerGlobalIdx] = val1;
    }
  }
}





/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    LOG_INFO("Starting bitonic sort with array size: %d", size);

    srand(time(NULL));

    DTYPE* arrCpu = (DTYPE*)malloc(size * sizeof(DTYPE));
    if (!arrCpu) {
        LOG_ERROR("Failed to allocate memory for CPU array");
        return 1;
    }

    LOG_STAGE("Generating random array");
    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }
    
    LOG_INFO("Original array:");
    printArray(arrCpu, size, "  ");

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// arCpu contains the input random array
// arrSortedGpu should contain the sorted array copied from GPU to CPU
int *arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));
if (!arrSortedGpu) {
    LOG_ERROR("Failed to allocate memory for GPU result array");
    free(arrCpu);
    return 1;
}

LOG_STAGE("Setting up GPU memory and data transfer");

// Transfer data (arr_cpu) to device
// Declare the device pointer
DTYPE* d_arr;

// Allocate memory on the device
LOG_DEBUG("Allocating GPU memory for %d elements", size);
CUDA_CHECK(cudaMalloc(&d_arr, size * sizeof(DTYPE)));

// Copy data from host to device (including padded elements)
LOG_DEBUG("Copying data from host to device");
CUDA_CHECK(cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice));
LOG_INFO("Data successfully copied to GPU");

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Perform bitonic sort on GPU using shared memory
LOG_STAGE("Starting shared memory bitonic sort on GPU");
int threadsPerBlock = 256;
int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
LOG_INFO("Launch configuration: %d blocks, %d threads per block", blocksPerGrid, threadsPerBlock);

// Calculate shared memory size needed
size_t sharedMemSize = threadsPerBlock * sizeof(DTYPE);
LOG_DEBUG("Shared memory size: %zu bytes", sharedMemSize);

// Use smart hybrid approach: shared memory for small arrays, global memory for large arrays
int step_count = 0;
if (size <= 256) {
    // Small arrays: Use shared memory kernel (benefits from shared memory)
    LOG_DEBUG("Using shared memory kernel for small array");
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            step_count++;
            
            // Launch shared memory kernel
            BitonicSort_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_arr, j, k, size);
            
            // Check for kernel launch errors
            cudaError_t launchError = cudaGetLastError();
            if (launchError != cudaSuccess) {
                LOG_ERROR("Shared memory kernel launch failed: %s", cudaGetErrorString(launchError));
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Print intermediate results for small arrays only
            if (size <= 32) {
                DTYPE* temp = (DTYPE*)malloc(size * sizeof(DTYPE));
                CUDA_CHECK(cudaMemcpy(temp, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
                printf("  After step %d: ", step_count);
                printArray(temp, size, "", size);
                free(temp);
            }
        }
    }
    LOG_INFO("Shared memory bitonic sort completed in %d steps", step_count);
} else {
    // Large arrays: Use global memory kernel (no shared memory conflicts)
    LOG_DEBUG("Using global memory kernel for large array");
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            step_count++;
            
            // Launch global memory kernel
            BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
            
            // Check for kernel launch errors
            cudaError_t launchError = cudaGetLastError();
            if (launchError != cudaSuccess) {
                LOG_ERROR("Global memory kernel launch failed: %s", cudaGetErrorString(launchError));
            }
            
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    LOG_INFO("Global memory bitonic sort completed in %d steps", step_count);
}

// Check for kernel launch errors
cudaError_t launchError = cudaGetLastError();
if (launchError != cudaSuccess) {
    LOG_ERROR("Shared memory kernel launch failed: %s", cudaGetErrorString(launchError));
}

CUDA_CHECK(cudaDeviceSynchronize());
LOG_DEBUG("Shared memory kernel completed");

// Print intermediate results for small arrays
if (size <= 32) {
    DTYPE* temp = (DTYPE*)malloc(size * sizeof(DTYPE));
    CUDA_CHECK(cudaMemcpy(temp, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    printf("  After shared memory sort: ");
    printArray(temp, size, "", size);
    free(temp);
}

LOG_INFO("Shared memory bitonic sort completed");

/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);

/* ==== DO NOT MODIFY CODE ABOVE THIS LINE ==== */

// Transfer sorted data back to host (copied to arrSortedGpu)
LOG_STAGE("Transferring sorted data back to host");
CUDA_CHECK(cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
LOG_INFO("Sorted array:");
printArray(arrSortedGpu, size, "  ");



/* ==== DO NOT MODIFY CODE BELOW THIS LINE ==== */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    LOG_STAGE("Performing CPU sort for comparison");
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    LOG_STAGE("Validating results");
    int match = 1;
    int first_mismatch = -1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            first_mismatch = i;
            LOG_ERROR("Mismatch at index %d: GPU=%d, CPU=%d", i, arrSortedGpu[i], arrCpu[i]);
            break;
        }
    }
    
    if (match) {
        LOG_INFO("Validation successful - GPU and CPU results match!");
    } else {
        LOG_ERROR("Validation failed - results don't match at index %d", first_mismatch);
        // Print arrays around the mismatch for debugging
        if (first_mismatch >= 0) {
            int start = (first_mismatch > 5) ? first_mismatch - 5 : 0;
            int end = (first_mismatch < size - 5) ? first_mismatch + 5 : size;
            LOG_DEBUG("GPU array around mismatch:");
            printSubArray(arrSortedGpu, start, end, "  ");
            LOG_DEBUG("CPU array around mismatch:");
            printSubArray(arrCpu, start, end, "  ");
        }
    }

    // Clean up GPU memory
    CUDA_CHECK(cudaFree(d_arr));
    
    free(arrCpu);
    free(arrSortedGpu);

    if (match) {
        LOG_INFO("FUNCTIONAL SUCCESS");
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    } else {
        LOG_ERROR("FUNCTIONAL FAIL");
        printf("\033[1;31mFUNCTIONAL FAIL\n\033[0m");
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




