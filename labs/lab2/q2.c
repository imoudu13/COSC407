#include <stdio.h>
#include <stdlib.h>

int* addVec2(int* A, int* B, int size);

int main() {
    const int FIFTY_MILLION = 50000000;
    int* A = calloc(FIFTY_MILLION, sizeof(int));

    if (A == NULL) {
        printf("Error with memory allocation for A.\n");
        return 1;
    }

    int* B = calloc(FIFTY_MILLION, sizeof(int));

    if (B == NULL) {
        printf("Error with memory allocation for B.\n");
        return 1;
    }

    int* C = addVec2(A, B, FIFTY_MILLION);

    for (int i = 0; i < 10; i++) {
        if (i < 10) {
            printf("C[%d] = %d\n", i, C[i]);
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}

int* addVec2(int* A, int* B, int size) {
    int* C = calloc(size, sizeof(int));

    if (C == NULL) {
        printf("Memory allocation of C failed\n");
    }

    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }

    return C;
}
