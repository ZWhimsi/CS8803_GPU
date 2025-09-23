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

// Logging macros
#define LOG_INFO(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_STAGE(stage) printf("\n[STAGE] %s\n", stage)

// Optimized global kernel with loop unrolling
__global__ void BitonicSort_global(int* data, int j, int k, int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread for better efficiency
    for (int i = tid; i < size; i += stride) {
        int partnerIdx = i ^ j;
        
        if (i < partnerIdx && partnerIdx < size) {
            bool ascending = (i & k) == 0;
            
            int val1 = data[i];
            int val2 = data[partnerIdx];
            
            if ((val1 > val2) == ascending) {
                data[i] = val2;
                data[partnerIdx] = val1;
            }
        }
    }
}

// Warp-optimized kernel using shuffle operations
__global__ void BitonicSort_warp(int* data, int j, int k, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid >= size) return;
    
    int val = data[tid];
    int lane_id = threadIdx.x & 31;
    int warp_id = tid >> 5;
    
    // For j < 32, we can use warp shuffles within each warp
    if (j < 32) {
        unsigned mask = 0xFFFFFFFF;
        int partner_val = __shfl_xor_sync(mask, val, j);
        int partner_tid = tid ^ j;
        
        if (partner_tid < size) {
            bool ascending = (tid & k) == 0;
            if ((val > partner_val) == ascending && (tid < partner_tid)) {
                val = partner_val;
            } else if ((val < partner_val) != ascending && (tid < partner_tid)) {
                // Do nothing, keep current value
            } else if (tid > partner_tid) {
                // Higher thread takes the other value
                bool partner_ascending = (partner_tid & k) == 0;
                if ((partner_val > val) == partner_ascending) {
                    val = partner_val;
                }
            }
            data[tid] = val;
        }
    }
}

// GPU Kernel for bitonic sort with shared memory
__global__ void BitonicSort_shared(int* data, int j, int k, int size){
    extern __shared__ DTYPE s_array[];
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Load data into shared memory with bounds checking
    if (tid < size) {
        s_array[threadIdx.x] = data[tid];
    } else {
        s_array[threadIdx.x] = INT_MAX;  // Padding for out-of-bounds
    }
    __syncthreads();
    
    // Only process if both elements are within bounds
    if (tid < size) {
        int partnerIdx = tid ^ j;
        
        if (partnerIdx < size && tid < partnerIdx) {
            int partnerBlockIdx = partnerIdx / blockDim.x;
            bool sameBlock = (partnerBlockIdx == blockIdx.x);
            
            if (sameBlock) {
                int partnerLocalIdx = partnerIdx % blockDim.x;
                bool ascending = (tid & k) == 0;
                
                int val1 = s_array[threadIdx.x];
                int val2 = s_array[partnerLocalIdx];
                
                if ((val1 > val2) == ascending) {
                    s_array[threadIdx.x] = val2;
                    s_array[partnerLocalIdx] = val1;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Write back to global memory
    if (tid < size) {
        data[tid] = s_array[threadIdx.x];
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
    
    // Allocate pinned memory for faster transfers
    int* h_arr_pinned = nullptr;
    int* h_result_pinned = nullptr;
    
    cudaError_t pinned_in = cudaMallocHost(&h_arr_pinned, size * sizeof(int));
    cudaError_t pinned_out = cudaMallocHost(&h_result_pinned, size * sizeof(int));
    
    bool use_pinned_input = (pinned_in == cudaSuccess);
    bool use_pinned_output = (pinned_out == cudaSuccess);
    
    if (use_pinned_input) {
        memcpy(h_arr_pinned, arrCpu, size * sizeof(int));
        LOG_DEBUG("Using pinned memory for input");
    } else {
        LOG_DEBUG("Using regular memory for input");
    }
    
    if (use_pinned_output) {
        LOG_DEBUG("Using pinned memory for output");
    } else {
        h_result_pinned = (int*)malloc(size * sizeof(int));
        LOG_DEBUG("Using regular memory for output");
    }

    LOG_STAGE("Setting up GPU memory and data transfer");
    
    // Allocate device memory
    DTYPE* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, size * sizeof(DTYPE)));

    // Initial data transfer
    LOG_DEBUG("Copying data from host to device");
    CUDA_CHECK(cudaMemcpy(d_arr, use_pinned_input ? h_arr_pinned : arrCpu, 
                          size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    LOG_INFO("Data successfully copied to GPU");

    /* ==== TIMING SECTION START ==== */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float h2dTime, kernelTime, d2hTime;

    // Measure H2D transfer
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_arr, use_pinned_input ? h_arr_pinned : arrCpu, 
                          size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    // Measure kernel execution
    cudaEventRecord(start);

    // Perform bitonic sort on GPU
    LOG_STAGE("Starting optimized bitonic sort on GPU");
    
    // OPTIMIZATION: Use 1024 threads for H100
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Limit grid size to avoid overhead
    blocksPerGrid = min(blocksPerGrid, 65535);
    
    LOG_INFO("Launch configuration: %d blocks, %d threads per block", blocksPerGrid, threadsPerBlock);
    
    size_t sharedMemSize = threadsPerBlock * sizeof(DTYPE);
    int kernel_launches = 0;
    
    // Main bitonic sort loop
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            
            if (j < 32) {
                // Use warp shuffle for very small j
                BitonicSort_warp<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
            } else if (j < threadsPerBlock && k <= 2048) {
                // Use shared memory when partners are likely in same block
                BitonicSort_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_arr, j, k, size);
            } else {
                // Use global memory for large j values
                BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
            }
            
            kernel_launches++;
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_INFO("Bitonic sort completed with %d kernel launches", kernel_launches);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // Measure D2H transfer
    cudaEventRecord(start);
    
    LOG_STAGE("Transferring sorted data back to host");
    CUDA_CHECK(cudaMemcpy(use_pinned_output ? h_result_pinned : arrCpu, d_arr, 
                          size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);
    /* ==== TIMING SECTION END ==== */

    // Show sorted array (only original elements, not padding)
    LOG_INFO("Sorted array:");
    printArray(use_pinned_output ? h_result_pinned : arrCpu, original_size, "  ");

    // CPU sort for validation
    LOG_STAGE("Performing CPU sort for comparison");
    int* arrCpuSorted = (int*)malloc(size * sizeof(int));
    memcpy(arrCpuSorted, arrCpu, size * sizeof(int));
    
    clock_t cpuStart = clock();
    bitonicSortCPU(arrCpuSorted, size);
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;

    // Validate results (only check original elements, not padding)
    LOG_STAGE("Validating results");
    bool valid = true;
    int* gpu_result = use_pinned_output ? h_result_pinned : arrCpu;
    
    for (int i = 0; i < original_size; i++) {
        if (gpu_result[i] != arrCpuSorted[i]) {
            LOG_ERROR("Mismatch at index %d: GPU=%d, CPU=%d", i, gpu_result[i], arrCpuSorted[i]);
            LOG_DEBUG("GPU array around mismatch:");
            for (int j = max(0, i-5); j < min(original_size, i+5); j++) {
                printf("  [%d]: GPU=%d, CPU=%d%s\n", j, gpu_result[j], arrCpuSorted[j], 
                       (j == i) ? " <-- MISMATCH" : "");
            }
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
        printf("- Input pinned memory: %s\n", use_pinned_input ? "YES" : "NO");
        printf("- Output pinned memory: %s\n", use_pinned_output ? "YES" : "NO");
        printf("- Kernel launches: %d\n", kernel_launches);
        printf("- Warp shuffles: Used for j < 32\n");
        printf("- Shared memory: Used for j < %d when k <= 2048\n", threadsPerBlock);
    } else {
        LOG_ERROR("FUNCTIONAL FAIL");
        printf("FUNCTIONAL FAIL\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    if (use_pinned_input) {
        CUDA_CHECK(cudaFreeHost(h_arr_pinned));
    }
    if (use_pinned_output) {
        CUDA_CHECK(cudaFreeHost(h_result_pinned));
    } else {
        free(h_result_pinned);
    }
    
    free(arrCpu);
    free(arrCpuSorted);
    
    return valid ? 0 : 1;
}
