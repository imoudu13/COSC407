#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

const int N = 1 << 24;  // this is 2 ^ 24, it's faster to compute it this way
const int THREADS_PER_BLOCK = 128;

// shared memory, more divergence
__global__ void v1(float* input, float* output) {
    __shared__ float sdata[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2 * s)) == 0 && tid + s < blockDim.x)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// shared memory, less divergence
__global__ void v2(float* input, float* output) {
    __shared__ float sdata[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// global memory only, more divergence
__global__ void v3(float* input, float* output) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int step = 1;
    while (step < blockDim.x) {
        if ((tid % (2 * step)) == 0 && i + step < N)
            input[i] += input[i + step];
        __syncthreads();
        step *= 2;
    }

    if (tid == 0) output[blockIdx.x] = input[blockIdx.x * blockDim.x];
}

// global memory only, less divergence
__global__ void v4(float* input, float* output) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < N)
            input[i] += input[i + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = input[blockIdx.x * blockDim.x];
}

// accepts the kernel function, name, and input and output
// I put this all in a function since there's so much repeatd things
void run(void (*kernel)(float*, float*), const char* name, float* input, float* output) {
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(output, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaEventRecord(start);

    kernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(input, output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    float result;
    cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%s result: %d, time: %d ms\n", name, result, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    float h_input[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand() % 256);
    }

    float* input;
    float* output;
    cudaMalloc(&input, N * sizeof(float));
    cudaMalloc(&output, N * sizeof(float));
    cudaMemcpy(input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    run(v1, "Version 1 (Shared + Divergence)", input, output);
    run(v2, "Version 2 (Shared + Less Divergence)", input, output);
    run(v3, "Version 3 (Global + Divergence)", input, output);
    run(v4, "Version 4 (Global + Less Divergence)", input, output);

    cudaFree(input);
    cudaFree(output);
    return 0;
}
