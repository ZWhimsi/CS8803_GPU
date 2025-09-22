#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("Testing basic CUDA functionality...\n\n");
    
    // 1. Check CUDA device
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    // 2. Set device
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("cudaSetDevice FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Successfully set device 0\n");
    
    // 3. Try small allocation
    void* d_ptr;
    size_t size = 1024; // Just 1KB
    printf("\nTrying to allocate 1KB on GPU... ");
    err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        printf("FAILED\n");
        printf("Error: %s (code %d)\n", cudaGetErrorString(err), err);
        
        // Try to reset
        printf("\nAttempting device reset...\n");
        cudaDeviceReset();
        
        // Try again after reset
        printf("Trying allocation again after reset... ");
        err = cudaMalloc(&d_ptr, size);
        if (err != cudaSuccess) {
            printf("STILL FAILED\n");
            printf("Error: %s (code %d)\n", cudaGetErrorString(err), err);
            return 1;
        }
    }
    
    printf("SUCCESS\n");
    
    // 4. Try larger allocations
    size_t test_sizes[] = {1024*1024, 10*1024*1024, 100*1024*1024, 512*1024*1024};
    const char* size_names[] = {"1MB", "10MB", "100MB", "512MB"};
    
    for (int i = 0; i < 4; i++) {
        printf("Allocating %s... ", size_names[i]);
        void* test_ptr;
        err = cudaMalloc(&test_ptr, test_sizes[i]);
        if (err == cudaSuccess) {
            printf("SUCCESS\n");
            cudaFree(test_ptr);
        } else {
            printf("FAILED (%s)\n", cudaGetErrorString(err));
            break;
        }
    }
    
    // Clean up
    if (d_ptr) cudaFree(d_ptr);
    
    printf("\nCUDA test completed.\n");
    return 0;
}
