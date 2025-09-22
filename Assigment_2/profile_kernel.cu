#include <cuda_profiler_api.h>

// Add this function to get detailed kernel metrics
void profileKernel() {
    // Warmup run
    int size = 134217728;
    int* d_arr;
    cudaMalloc(&d_arr, size * sizeof(int));
    
    // Initialize with test data
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Profile specific kernels
    cudaProfilerStart();
    
    // Test different j values to see memory access patterns
    int test_j_values[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    
    for (int i = 0; i < 13; i++) {
        int j = test_j_values[i];
        int k = j * 2;
        
        // Time the kernel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        BitonicSort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, j, k, size);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("j=%d: %.3f ms\n", j, milliseconds);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaProfilerStop();
    cudaFree(d_arr);
}
