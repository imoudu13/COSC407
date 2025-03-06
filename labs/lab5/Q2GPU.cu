#include <stdio.h>

__global__ void printElements(double *arr, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && (idx < 5 || idx >= size - 5)) { // First 5 or last 5 elements
        printf("Element at index %d: %.7f\n", idx, arr[idx]);
    }
}

int main() {
    const int n = 10000000; // 10 million
    double *h_arr = new double[n];
    double *d_arr;

    // Initialize array with values
    for (int i = 0; i < n ; i++) {
        h_arr[i] = (double)i / n;
    }

    cudaMalloc((void**)&d_arr, n * sizeof(double));
    cudaMemcpy(d_arr, h_arr, n * sizeof(double), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printElements<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n);
    cudaDeviceSynchronize();

    cudaFree(d_arr);
    delete[] h_arr;
    return 0;
}
