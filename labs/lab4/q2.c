#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define NRA 20
#define NCA 30
#define NCB 10

void printMatrix(int rows, int cols, int matrix[rows][cols], const char *name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void parallelMatrixMultiplication(int rowsA, int columnsB, int columnsA, int A[rowsA][columnsA], int B[columnsA][columnsB], int C[rowsA][columnsB]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columnsB; j++) {
            C[i][j] = 0;
            for (int k = 0; k < columnsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void initializeMatrix(int rows, int columns, int matrix[rows][columns]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = 1 + rand() % 100; // random number between 1-100
        }
    }
}

int main() {
    int A[NRA][NCA]; // A (20x30)
    int B[NCA][NCB]; // B (30x10)
    int C[NRA][NCB]; // C (20x10) - result of A * B

    initializeMatrix(NRA, NCA, A);
    initializeMatrix(NCA, NCB, B);

    parallelMatrixMultiplication(NRA, NCB, NCA, A, B, C);

    printMatrix(NRA, NCA, A, "A");
    printMatrix(NCA, NCB, B, "B");
    printMatrix(NRA, NCB, C, "C (Result of A * B)");

    return 0;
}
