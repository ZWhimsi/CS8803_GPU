// Performance analysis kernel to identify bottlenecks
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DTYPE int
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(1); \
    } \
} while(0)

// Test kernel with different memory access patterns
__global__ void testMemoryAccess(int* data, int stride, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;
    
    int partner = tid ^ stride;
    if (partner < size && tid < partner) {
        int val1 = data[tid];
        int val2 = data[partner];
        if (val1 > val2) {
            data[tid] = val2;
            data[partner] = val1;
        }
    }
}

// Analyze memory access patterns for different stride values
void analyzeMemoryPatterns() {
    const int size = 134217728; // 128M elements
    const int iterations = 10;
    
    int* d_data;
    cudaMalloc(&d_data, size * sizeof(int));
    
    // Test different stride patterns
    int strides[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int num_strides = sizeof(strides) / sizeof(strides[0]);
    
    printf("\nMemory Access Pattern Analysis:\n");
    printf("Stride\tTime(ms)\tGB/s\tEfficiency\n");
    printf("------\t--------\t----\t----------\n");
    
    for (int i = 0; i < num_strides; i++) {
        int stride = strides[i];
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
        // Warmup
        testMemoryAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, stride, size);
        cudaDeviceSynchronize();
        
        // Time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int j = 0; j < iterations; j++) {
            testMemoryAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, stride, size);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        float avg_time = milliseconds / iterations;
        
        // Calculate bandwidth (each thread reads 2 ints and writes 2 ints = 16 bytes)
        float bytes_accessed = (size / 2.0f) * 16; // Only half threads do work
        float bandwidth_gbs = (bytes_accessed / 1e9) / (avg_time / 1000.0f);
        
        // H100 peak bandwidth is ~3.35 TB/s
        float efficiency = (bandwidth_gbs / 3350.0f) * 100.0f;
        
        printf("%d\t%.3f\t\t%.1f\t%.1f%%\n", stride, avg_time, bandwidth_gbs, efficiency);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFree(d_data);
}

// Test different thread block sizes
void analyzeBlockSizes() {
    const int size = 134217728;
    int* d_data;
    cudaMalloc(&d_data, size * sizeof(int));
    
    printf("\nBlock Size Analysis:\n");
    printf("Threads\tBlocks\t\tTime(ms)\n");
    printf("-------\t------\t\t--------\n");
    
    int block_sizes[] = {64, 128, 256, 512, 1024};
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int threadsPerBlock = block_sizes[i];
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        testMemoryAccess<<<blocksPerGrid, threadsPerBlock>>>(d_data, 1024, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("%d\t%d\t\t%.3f\n", threadsPerBlock, blocksPerGrid, milliseconds);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFree(d_data);
}

// Analyze memory transfer optimizations
void analyzeMemoryTransfers() {
    const int size = 134217728; // 128M elements
    size_t bytes = size * sizeof(int);
    
    // Test regular memory
    int* h_data = (int*)malloc(bytes);
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        h_data[i] = rand() % 1000;
    }
    
    printf("Transfer Type\t\t\tH2D (ms)\tD2H (ms)\tBandwidth (GB/s)\n");
    printf("-------------\t\t\t--------\t--------\t----------------\n");
    
    // Test 1: Regular memory
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float h2d_regular = 0;
    cudaEventElapsedTime(&h2d_regular, start, stop);
    
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2h_regular = 0;
    cudaEventElapsedTime(&d2h_regular, start, stop);
    
    printf("Regular Memory\t\t\t%.2f\t\t%.2f\t\t%.1f / %.1f\n", 
           h2d_regular, d2h_regular, 
           (bytes/1e9)/(h2d_regular/1000), (bytes/1e9)/(d2h_regular/1000));
    
    // Test 2: Pinned memory
    int* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    memcpy(h_pinned, h_data, bytes);
    
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float h2d_pinned = 0;
    cudaEventElapsedTime(&h2d_pinned, start, stop);
    
    cudaEventRecord(start);
    cudaMemcpy(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2h_pinned = 0;
    cudaEventElapsedTime(&d2h_pinned, start, stop);
    
    printf("Pinned Memory\t\t\t%.2f\t\t%.2f\t\t%.1f / %.1f\n", 
           h2d_pinned, d2h_pinned,
           (bytes/1e9)/(h2d_pinned/1000), (bytes/1e9)/(d2h_pinned/1000));
    
    // Test 3: Async with streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEventRecord(start);
    cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float h2d_async = 0;
    cudaEventElapsedTime(&h2d_async, start, stop);
    
    cudaEventRecord(start);
    cudaMemcpyAsync(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float d2h_async = 0;
    cudaEventElapsedTime(&d2h_async, start, stop);
    
    printf("Pinned + Async\t\t\t%.2f\t\t%.2f\t\t%.1f / %.1f\n", 
           h2d_async, d2h_async,
           (bytes/1e9)/(h2d_async/1000), (bytes/1e9)/(d2h_async/1000));
    
    printf("\nSpeedup vs Regular Memory:\n");
    printf("Pinned Memory: %.2fx H2D, %.2fx D2H\n", h2d_regular/h2d_pinned, d2h_regular/d2h_pinned);
    printf("Pinned + Async: %.2fx H2D, %.2fx D2H\n", h2d_regular/h2d_async, d2h_regular/d2h_async);
    
    // Cleanup
    free(h_data);
    cudaFreeHost(h_pinned);
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Test kernel for warp efficiency
__global__ void testWarpDivergence(int* data, int j, int k, int size, int* active_count) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (tid >= size) return;
    
    int partner = tid ^ j;
    bool is_active = (tid < partner) && (partner < size);
    
    // Count active threads per warp
    unsigned mask = __ballot_sync(0xffffffff, is_active);
    if (lane_id == 0) {
        atomicAdd(&active_count[warp_id], __popc(mask));
    }
}

// Analyze warp efficiency for different patterns
void analyzeWarpEfficiency() {
    const int size = 134217728;
    int* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(int)));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int total_warps = (size + 31) / 32;
    
    int* d_active_count;
    CUDA_CHECK(cudaMalloc(&d_active_count, total_warps * sizeof(int)));
    
    printf("j\tAvg Active Threads/Warp\tEfficiency\n");
    printf("---\t----------------------\t----------\n");
    
    int j_values[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    
    for (int i = 0; i < sizeof(j_values)/sizeof(j_values[0]); i++) {
        int j = j_values[i];
        if (j >= size) break;
        
        int k = j * 2;
        
        // Reset counter
        CUDA_CHECK(cudaMemset(d_active_count, 0, total_warps * sizeof(int)));
        
        // Run kernel
        testWarpDivergence<<<blocksPerGrid, threadsPerBlock>>>(d_data, j, k, size, d_active_count);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Get results
        int* h_active_count = (int*)malloc(total_warps * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_active_count, d_active_count, total_warps * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Calculate average
        long long total_active = 0;
        int valid_warps = 0;
        for (int w = 0; w < total_warps; w++) {
            if (h_active_count[w] > 0) {
                total_active += h_active_count[w];
                valid_warps++;
            }
        }
        
        float avg_active = (valid_warps > 0) ? (float)total_active / valid_warps : 0;
        float efficiency = avg_active / 32.0f * 100.0f;
        
        printf("%d\t%.2f\t\t\t%.1f%%\n", j, avg_active, efficiency);
        
        free(h_active_count);
    }
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_active_count));
}

int main() {
    printf("GPU Performance Analysis for Bitonic Sort\n");
    printf("=========================================\n");
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.0f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("L2 Cache Size: %.1f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    
    analyzeMemoryPatterns();
    analyzeBlockSizes();
    
    // Additional analysis for memory transfer optimization
    printf("\n\nMemory Transfer Analysis:\n");
    printf("========================\n");
    analyzeMemoryTransfers();
    
    // Warp efficiency analysis
    printf("\n\nWarp Efficiency Analysis:\n");
    printf("========================\n");
    analyzeWarpEfficiency();
    
    return 0;
}
