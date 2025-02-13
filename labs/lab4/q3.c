#include <stdio.h>
#include <omp.h>

int main() {
    int n = 10; // Example size, you can change it
    int a[n];

    // Parallel computation of the sequence
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        a[i] = (i * (i + 1)) / 2;
    }

    // Print the results
    printf("Computed array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    return 0;
}
