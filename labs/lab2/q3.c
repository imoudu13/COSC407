#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    int numThreads, tid;

    #pragma omp parallel private(tid) shared(numThreads) 
    {
        tid = omp_get_thread_num();
        
        #pragma omp critical
        {
            printf("Hello World from thread number %d\n", tid);
        }

        if (tid == 0) {
            numThreads = omp_get_num_threads();
            printf("Number of threads is %d\n", numThreads);
        }
    }

    return 0;
}
