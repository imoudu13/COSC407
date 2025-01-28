#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define MILLION 1000000
int* vecCreate(int size);
int* vecCreateOpenMP(int size, int num_thread);

int main() {
    const int SIZE = 50 * MILLION;

    clock_t start_seq = clock();
    int* sequential = vecCreate(SIZE);
    clock_t end_seq = clock();

    if (sequential != NULL) {
        printf("Sequential: Last element: %d\n", sequential[SIZE - 1]);
        double time_taken = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;
        printf("Sequential: Time taken: %.2f seconds\n\n", time_taken);
        free(sequential);
    }

    int num_threads = 4;

    clock_t start_par = omp_get_wtime();
    int* parallel = vecCreateOpenMP(SIZE, num_threads);
    clock_t end_par = omp_get_wtime();

    if (parallel != NULL) {
        printf("Parallel: Last element: %d\n", parallel[SIZE - 1]);
        double time_taken = (double)(end_par - start_par) / CLOCKS_PER_SEC;
        printf("Parallel: Time taken: %.2f seconds\n", time_taken);
        free(parallel);
    }

    return 0;
}

int* vecCreate(const int SIZE) {
    int* A = malloc(SIZE * sizeof(int));

    if (A == NULL) {
        printf("There was a problem allocating memory to create the array of size: %d\n", SIZE);
        return NULL;
    }

    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
    }

    return A;
}

int* vecCreateOpenMP(const int SIZE, int thread_count) {
    printf("Using OpenMP with 3 threads: \n");
    if (SIZE % thread_count != 0) {
        printf("Error: Vector size must be divisible by the number of threads.\n");
        return NULL;
    }

    int* A = malloc(SIZE * sizeof(int));

    if (A == NULL) {
        printf("There was a problem allocating memory to create the array of size: %d\n", SIZE);
        return NULL;
    }

    # pragma omp parallel num_threads(thread_count)
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = SIZE / thread_count;
        int start = chunk_size * thread_id;
        int end = start + chunk_size;

        for (int i = start; i < end; i++) {
            A[i] = i;
        }
    }
    
    return A;
}

/*
Sequential: Last element: 49999999
Sequential: Time taken: 0.16 seconds
Parallel: Last element: 49999999
Parallel: Time taken: 0.00 seconds

Sequential: Last element: 49999999
Sequential: Time taken: 0.15 seconds
Error: Vector size must be divisible by the number of threads.
*/