```
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 1024

// Kernel function: computes square of each input element
__global__ void gpu_sqr(int *d_in, int *d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

// CPU version (already provided)
void cpu_sqr(int *data_in, int *data_out, int size) {
    for (int i = 0; i < size; ++i) {
        data_out[i] = data_in[i] * data_in[i];
    }
}

// Error checking macro
#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

int main() {
    const int N = 10000;
    int *h_in, *h_out;     // Host pointers
    int *d_in, *d_out;     // Device pointers
    dim3 grid;

    // Allocate host memory
    h_in = (int*)malloc(N * sizeof(int));
    h_out = (int*)malloc(N * sizeof(int));

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // 1. Allocate memory on GPU
    CHECK_CUDA(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // 2. Copy data to GPU
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // 3. Calculate grid size
    grid.x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 4. Launch kernel
    gpu_sqr<<<grid, BLOCK_SIZE>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch error

    // 5. Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // (Optional) Verify correctness
    for (int i = 0; i < 10; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    // 6. Free all resources
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
```