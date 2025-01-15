#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void getC(int* A, int* B, const int SIZE);
void print_cpu_time(clock_t start, clock_t end);

int main() {
    const int ONE_MILLION = 1000000;
    const int TEN_MILLION = 10000000;
    const int FIFTY_MILLION = 50000000;
    const int TWO_HUNDRED_MILLION = 200000000;

    clock_t start, end;

    // one milllion
    int* ONE_MILLION_A = (int*) malloc(ONE_MILLION * sizeof(int));
    int* ONE_MILLION_B = (int*) malloc(ONE_MILLION * sizeof(int));

    start = clock();

    getC(ONE_MILLION_A, ONE_MILLION_B, ONE_MILLION);

    end = clock();

    free(ONE_MILLION_A);
    free(ONE_MILLION_B);

    print_cpu_time(start, end);

    // 10 million 
    int* TEN_MILLION_A = (int*) malloc(TEN_MILLION * sizeof(int));
    int* TEN_MILLION_B = (int*) malloc(TEN_MILLION * sizeof(int));

    free(TEN_MILLION_A);
    free(TEN_MILLION_B);
    
    start = clock();

    getC(TEN_MILLION_A, TEN_MILLION_B, ONE_MILLION);

    end = clock();

    print_cpu_time(start, end);

    // 50 million
    int* FIFTY_MILLION_A = (int*) malloc(FIFTY_MILLION * sizeof(int));
    int* FIFTY_MILLION_B = (int*) malloc(FIFTY_MILLION * sizeof(int));

    start = clock();

    getC(FIFTY_MILLION_A, FIFTY_MILLION_B, FIFTY_MILLION);

    end = clock();

    
    free(FIFTY_MILLION_A);
    free(FIFTY_MILLION_B);

    print_cpu_time(start, end);

    // 200 million
    int* TWO_HUNDRED_MILLION_A = (int*) malloc(TWO_HUNDRED_MILLION * sizeof(int));
    int* TWO_HUNDRED_MILLION_B = (int*) malloc(TWO_HUNDRED_MILLION * sizeof(int));

    start = clock();

    getC(TWO_HUNDRED_MILLION_A, TWO_HUNDRED_MILLION_B, TWO_HUNDRED_MILLION);

    end = clock();

    free(TWO_HUNDRED_MILLION_A);
    free(TWO_HUNDRED_MILLION_B);

    print_cpu_time(start, end);

    return 0;
}

void getC(int* A, int* B, const int SIZE) {
    printf("Starting computation for size: %d\n", SIZE);

    int* C = (int*) malloc(SIZE * sizeof(int));

    int sum = 0;

    for (int i = 0; i < SIZE; i++) {
        int tempA = i * 3;
        int tempB = -i * 3;

        A[i] = tempA;
        B[i] = tempB;

        C[i] = tempA + tempB;

        sum += C[i];
    }

    printf("The sum of size: %d is %d\n", SIZE, sum);

    free(C);
}

void print_cpu_time(clock_t start, clock_t end) {
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;  // Convert to seconds
    printf("CPU time used: %f seconds\n", cpu_time_used);
}