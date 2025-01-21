#include <stdio.h>
#include <stdlib.h>

void addVec(int *C, int *A, int *B, int size);

int main() {
    const int FIFTY_MILLION = 50000000;

    int *A = calloc(FIFTY_MILLION, sizeof(int));
    if (A == NULL) {
        fprintf(stderr, "Memory allocation failed for A\n");
        return 1;
    }

    int *B = calloc(FIFTY_MILLION, sizeof(int));
    if (B == NULL) {
        fprintf(stderr, "Memory allocation failed for B\n");
        free(A); // Clean up previously allocated memory
        return 1;
    }

    int *C = calloc(FIFTY_MILLION, sizeof(int));
    if (C == NULL) {
        fprintf(stderr, "Memory allocation failed for C\n");
        free(A);
        free(B);
        return 1;
    }

    addVec(C, A, B, FIFTY_MILLION);

    free(A);
    free(B);
    free(C);

    return 0;
}

void addVec(int *C, int *A, int *B, int size){
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];

        if (i < 10) {
            printf("C[%d] = %d\n", i, C[i]);
        }
    }
}