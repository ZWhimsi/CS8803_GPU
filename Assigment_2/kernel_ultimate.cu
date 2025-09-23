#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <string.h>
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

#define LOG_INFO(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_STAGE(stage) printf("\n[STAGE] %s\n", stage)

// ULTIMATE OPTIMIZATION: Vectorized memory access using int4
__global__ void BitonicSort_vectorized(int* data, int j, int k, int size) {
    int tid = (blockDim.x * blockIdx.x + threadIdx.x) * 4; // Process 4 elements per thread
    
    if (tid >= size) return;
    
    // Load 4 elements at once
    int4 vals;
    if (tid + 3 < size) {
        vals = *reinterpret_cast<int4*>(&data[tid]);
    } else {
        // Handle edge case
        vals.x = (tid < size) ? data[tid] : INT_MAX;
        vals.y = (tid + 1 < size) ? data[tid + 1] : INT_MAX;
        vals.z = (tid + 2 < size) ? data[tid + 2] : INT_MAX;
        vals.w = (tid + 3 < size) ? data[tid + 3] : INT_MAX;
    }
    
    // Process each element
    int values[4] = {vals.x, vals.y, vals.z, vals.w};
    int partners[4];
    bool changed = false;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i;
        if (idx < size) {
            int partnerIdx = idx ^ j;
            if (idx < partnerIdx && partnerIdx < size) {
                partners[i] = data[partnerIdx];
                bool ascending = (idx & k) == 0;
                if ((values[i] > partners[i]) == ascending) {
                    // Swap needed
                    data[partnerIdx] = values[i];
                    values[i] = partners[i];
                    changed = true;
                }
            }
        }
    }
    
    // Write back if changed
    if (changed && tid + 3 < size) {
        vals.x = values[0];
        vals.y = values[1];
        vals.z = values[2];
        vals.w = values[3];
        *reinterpret_cast<int4*>(&data[tid]) = vals;
    } else if (changed) {
        // Handle edge case
        for (int i = 0; i < 4 && tid + i < size; i++) {
            data[tid + i] = values[i];
        }
    }
}

// Highly optimized global kernel with prefetching
__global__ void BitonicSort_optimized(int* data, int j, int k, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread with prefetching
    for (int base = tid; base < size; base += stride * 4) {
        #pragma unroll 4
        for (int offset = 0; offset < 4 && base + offset * stride < size; offset++) {
            int i = base + offset * stride;
            int partnerIdx = i ^ j;
            
            if (i < partnerIdx && partnerIdx < size) {
                bool ascending = (i & k) == 0;
                
                // Prefetch next iteration's data
                if (offset < 3 && base + (offset + 1) * stride < size) {
                    int next_i = base + (offset + 1) * stride;
                    int next_partner = next_i ^ j;
                    if (next_partner < size) {
                        volatile int prefetch1 = data[next_i];
                        volatile int prefetch2 = data[next_partner];
                    }
                }
                
                int val1 = data[i];
                int val2 = data[partnerIdx];
                
                if ((val1 > val2) == ascending) {
                    data[i] = val2;
                    data[partnerIdx] = val1;
                }
            }
        }
    }
}

// Standard kernels
__global__ void BitonicSort_global(int* data, int j, int k, int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size) return;
    
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

__global__ void BitonicSort_warp(int* data, int j, int k, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid >= size) return;
    
    int val = data[tid];
    
    if (j < 32) {
        unsigned mask = 0xFFFFFFFF;
        int partner_val = __shfl_xor_sync(mask, val, j);
        int partner_tid = tid ^ j;
        
        if (partner_tid < size) {
            bool ascending = (tid & k) == 0;
            if ((val > partner_val) == ascending && (tid < partner_tid)) {
                val = partner_val;
            } else if ((val < partner_val) != ascending && (tid < partner_tid)) {
                // Keep current value
            } else if (tid > partner_tid) {
                bool partner_ascending = (partner_tid & k) == 0;
                if ((partner_val > val) == partner_ascending) {
                    val = partner_val;
                }
            }
            data[tid] = val;
        }
    }
}

__global__ void BitonicSort_shared(int* data, int j, int k, int size){
    extern __shared__ DTYPE s_array[];
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < size) {
        s_array[threadIdx.x] = data[tid];
    } else {
        s_array[threadIdx.x] = INT_MAX;
    }
    __syncthreads();
    
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
    
    if (tid < size) {
        data[tid] = s_array[threadIdx.x];
    }
}

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
    
    int size = nextPowerOfTwo(original_size);
    bool needs_padding = (size != original_size);
    
    if (needs_padding) {
        LOG_INFO("Padding array from %d to %d elements (next power of 2)", original_size, size);
    }
    
    // Allocate host memory
    int* arrCpu = (int*)malloc(size * sizeof(int));
    if (!arrCpu) {
        LOG_ERROR("Failed to allocate memory");
        return 1;
    }
    
    // Initialize array
    LOG_STAGE("Generating random array");
    srand(time(NULL));
    for (int i = 0; i < original_size; i++) {
        arrCpu[i] = rand() % 1000;
    }
    
    for (int i = original_size; i < size; i++) {
        arrCpu[i] = INT_MAX;
    }
    
    LOG_INFO("Original array:");
    printArray(arrCpu, original_size, "  ");
    
    // Use pinned memory for optimal transfers
    int* h_arr_pinned = nullptr;
    int* h_result_pinned = nullptr;
    
    // Create CUDA streams for overlapping
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    cudaError_t pinned_in = cudaMallocHost(&h_arr_pinned, size * sizeof(int));
    cudaError_t pinned_out = cudaMallocHost(&h_result_pinned, size * sizeof(int));
    
    bool use_pinned = (pinned_in == cudaSuccess && pinned_out == cudaSuccess);
    
    if (use_pinned) {
        memcpy(h_arr_pinned, arrCpu, size * sizeof(int));
        LOG_DEBUG("Using pinned memory with async transfers");
    } else {
        h_arr_pinned = arrCpu;
        h_result_pinned = (int*)malloc(size * sizeof(int));
        LOG_DEBUG("Using regular memory");
    }

    LOG_STAGE("Setting up GPU memory");
    
    DTYPE* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, size * sizeof(DTYPE)));

    // Warm up GPU
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr_pinned, size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    BitonicSort_global<<<1, 32>>>(d_arr, 1, 2, 32);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ==== TIMING START ==== */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float h2dTime, kernelTime, d2hTime;

    // H2D Transfer
    cudaEventRecord(start);
    if (use_pinned) {
        CUDA_CHECK(cudaMemcpyAsync(d_arr, h_arr_pinned, size * sizeof(DTYPE), 
                                    cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream1));
    } else {
        CUDA_CHECK(cudaMemcpy(d_arr, h_arr_pinned, size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    // Kernel execution
    cudaEventRecord(start);

    LOG_STAGE("Running ultimate optimized bitonic sort");
    
    // ULTIMATE CONFIGURATION: Balance between occupancy and efficiency
    int threadsPerBlock = 512;  // Sweet spot for H100
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // For vectorized kernel, adjust grid size
    int vectorizedThreadsPerBlock = 512;
    int vectorizedBlocksPerGrid = (size / 4 + vectorizedThreadsPerBlock - 1) / vectorizedThreadsPerBlock;
    
    blocksPerGrid = min(blocksPerGrid, 32768);
    vectorizedBlocksPerGrid = min(vectorizedBlocksPerGrid, 32768);
    
    LOG_INFO("Launch config: %d blocks, %d threads/block", blocksPerGrid, threadsPerBlock);
    
    size_t sharedMemSize = threadsPerBlock * sizeof(DTYPE);
    int kernel_launches = 0;
    
    // Main sorting loop with aggressive optimizations
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            
            if (j < 32) {
                // Warp shuffle for smallest j
                BitonicSort_warp<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_arr, j, k, size);
            } else if (j >= 128 && (j & (j-1)) == 0) {
                // Use vectorized kernel for power-of-2 j values >= 128
                BitonicSort_vectorized<<<vectorizedBlocksPerGrid, vectorizedThreadsPerBlock, 0, stream1>>>(d_arr, j, k, size);
            } else if (j < threadsPerBlock && k <= 2048) {
                // Shared memory for medium-sized patterns
                BitonicSort_shared<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream1>>>(d_arr, j, k, size);
            } else {
                // Optimized global kernel
                BitonicSort_optimized<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_arr, j, k, size);
            }
            
            kernel_launches++;
        }
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    LOG_INFO("Sort completed with %d kernel launches", kernel_launches);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // D2H Transfer
    cudaEventRecord(start);
    
    if (use_pinned) {
        CUDA_CHECK(cudaMemcpyAsync(h_result_pinned, d_arr, size * sizeof(DTYPE), 
                                   cudaMemcpyDeviceToHost, stream2));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
    } else {
        CUDA_CHECK(cudaMemcpy(h_result_pinned, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);
    /* ==== TIMING END ==== */

    LOG_INFO("Sorted array:");
    printArray(h_result_pinned, original_size, "  ");

    // Validation
    LOG_STAGE("Validating results");
    int* arrCpuSorted = (int*)malloc(size * sizeof(int));
    memcpy(arrCpuSorted, arrCpu, size * sizeof(int));
    
    clock_t cpuStart = clock();
    bitonicSortCPU(arrCpuSorted, size);
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000.0;

    bool valid = true;
    for (int i = 0; i < original_size; i++) {
        if (h_result_pinned[i] != arrCpuSorted[i]) {
            LOG_ERROR("Mismatch at index %d: GPU=%d, CPU=%d", i, h_result_pinned[i], arrCpuSorted[i]);
            valid = false;
            break;
        }
    }
    
    if (valid) {
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
        
        printf("\nUltimate Optimizations:\n");
        printf("- Vectorized memory access (int4)\n");
        printf("- Pinned memory: %s\n", use_pinned ? "YES" : "NO");
        printf("- Async transfers: %s\n", use_pinned ? "YES" : "NO");
        printf("- Kernel launches: %d\n", kernel_launches);
        printf("- Multi-element processing per thread\n");
        printf("- Prefetching in optimized kernel\n");
    } else {
        LOG_ERROR("FUNCTIONAL FAIL");
        printf("FUNCTIONAL FAIL\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_arr));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    
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
