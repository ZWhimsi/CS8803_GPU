#include <stdio.h>
#include <cuda_runtime.h>

void checkPinnedMemoryCapabilities() {
    printf("=== CUDA Pinned Memory Diagnostics ===\n\n");
    
    // 1. Check CUDA version and device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Can Map Host Memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
    }
    
    printf("\n=== Testing Pinned Memory Allocation ===\n");
    
    // 2. Try different sizes of pinned memory
    size_t test_sizes[] = {
        1 * 1024 * 1024,        // 1 MB
        10 * 1024 * 1024,       // 10 MB
        100 * 1024 * 1024,      // 100 MB
        256 * 1024 * 1024,      // 256 MB
        512 * 1024 * 1024,      // 512 MB
        1024 * 1024 * 1024      // 1 GB
    };
    
    for (int i = 0; i < 6; i++) {
        size_t size = test_sizes[i];
        void* ptr;
        
        printf("\nTrying to allocate %.2f MB of pinned memory... ", size / (1024.0 * 1024.0));
        fflush(stdout);
        
        cudaError_t result = cudaMallocHost(&ptr, size);
        
        if (result == cudaSuccess) {
            printf("SUCCESS\n");
            
            // Test if we can actually use it
            memset(ptr, 0, size);
            
            // Free it
            cudaFreeHost(ptr);
        } else {
            printf("FAILED\n");
            printf("  Error code: %d\n", result);
            printf("  Error string: %s\n", cudaGetErrorString(result));
            printf("  This is the maximum pinned memory size: %.2f MB\n", 
                   (i > 0 ? test_sizes[i-1] : 0) / (1024.0 * 1024.0));
            break;
        }
    }
    
    // 3. Check current memory status
    printf("\n=== Current Memory Status ===\n");
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Free Memory: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
    printf("GPU Total Memory: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    
    // 4. Try alternative pinned memory flags
    printf("\n=== Testing Alternative Allocation Methods ===\n");
    
    void* test_ptr;
    size_t test_size = 100 * 1024 * 1024; // 100MB
    
    // Try with different flags
    printf("Testing cudaHostAlloc with cudaHostAllocDefault... ");
    cudaError_t result = cudaHostAlloc(&test_ptr, test_size, cudaHostAllocDefault);
    if (result == cudaSuccess) {
        printf("SUCCESS\n");
        cudaFreeHost(test_ptr);
    } else {
        printf("FAILED (%s)\n", cudaGetErrorString(result));
    }
    
    printf("Testing cudaHostAlloc with cudaHostAllocPortable... ");
    result = cudaHostAlloc(&test_ptr, test_size, cudaHostAllocPortable);
    if (result == cudaSuccess) {
        printf("SUCCESS\n");
        cudaFreeHost(test_ptr);
    } else {
        printf("FAILED (%s)\n", cudaGetErrorString(result));
    }
    
    printf("Testing cudaHostAlloc with cudaHostAllocMapped... ");
    result = cudaHostAlloc(&test_ptr, test_size, cudaHostAllocMapped);
    if (result == cudaSuccess) {
        printf("SUCCESS\n");
        cudaFreeHost(test_ptr);
    } else {
        printf("FAILED (%s)\n", cudaGetErrorString(result));
    }
    
    printf("\n=== Recommendations ===\n");
    printf("If pinned memory allocation fails:\n");
    printf("1. Check system limits: ulimit -l (should show 'unlimited' or a large value)\n");
    printf("2. Try: sudo nvidia-smi -pm 1 (enable persistence mode)\n");
    printf("3. Restart GPU: sudo nvidia-smi -r\n");
    printf("4. Use regular memory as fallback (as implemented in kernel_optimized.cu)\n");
}

int main() {
    checkPinnedMemoryCapabilities();
    return 0;
}
