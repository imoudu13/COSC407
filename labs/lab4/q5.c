#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define N 10000 

void count_sort_serial(int a[], int n) {
    int i, j, count;
    int* temp = malloc(n * sizeof(int));

    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++) {
            if (a[j] < a[i] || (a[j] == a[i] && j < i))
                count++;
        }
        temp[count] = a[i];
    }
    memcpy(a, temp, n * sizeof(int));
    free(temp);
}

void count_sort_parallel(int a[], int n) {
    int i, j, count;
    int* temp = malloc(n * sizeof(int));

    #pragma omp parallel for private(i, j, count) shared(a, temp) 
    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++) {
            if (a[j] < a[i] || (a[j] == a[i] && j < i))
                count++;
        }
        #pragma omp critical
        temp[count] = a[i];
    }

    memcpy(a, temp, n * sizeof(int));
    free(temp);
}

void generate_array(int a[], int n) {
    for (int i = 0; i < n; i++)
        a[i] = rand() % 1000;
}

int main() {
    int *a_serial = malloc(N * sizeof(int));
    int *a_parallel = malloc(N * sizeof(int));

    srand(time(NULL));
    generate_array(a_serial, N);
    memcpy(a_parallel, a_serial, N * sizeof(int));

    double start_time = omp_get_wtime();
    count_sort_serial(a_serial, N);
    double end_time = omp_get_wtime();
    printf("Serial Count Sort Time: %f seconds\n", end_time - start_time);

    omp_set_num_threads(8); 
    start_time = omp_get_wtime();
    count_sort_parallel(a_parallel, N);
    end_time = omp_get_wtime();
    printf("Parallel Count Sort Time: %f seconds\n", end_time - start_time);
    
    // free arrays
    free(a_serial);
    free(a_parallel);
    
    return 0;
}


/* 
a)
a[] and temp[] should be shared otherwise the count, and i and j should the private
if you look at count_sort_parallel you can see the implementation

b)
no there is no loop carried dependency, there is a race condition with temp[] since it's shared multiple threads my write to the same location

*/


//Serial Count Sort Time: 0.290000 seconds
//Parallel Count Sort Time: 0.056000 seconds