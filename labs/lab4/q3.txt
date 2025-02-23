#include <stdio.h>
#include <omp.h>

#define N 10

int main() {
    int a[N];
    
    // using the ordered clause fixes the loop carried dependeny
    // you could also try reworking the algorithm 
    a[0] = 0;
    #pragma omp parallel for ordered
    for (int i = 1; i < N; i++) {
        #pragma omp ordered
        a[i] = a[i - 1] + i;
    }

    // Print results
    for (int i = 0; i < N; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    return 0;
}
