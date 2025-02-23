#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "my_rand.h"

int main() {
    long long int number_of_tosses, number_in_circle = 0;
    double x, y, distance_squared, pi_estimate;

    printf("Enter the total number of tosses: ");
    scanf("%lld", &number_of_tosses);

    #pragma omp parallel 
    {
        int thread_id = omp_get_thread_num();  
        unsigned int seed = time(NULL) + thread_id;
        long long int local_count = 0;

        #pragma omp for reduction(+:number_in_circle)
        for (long long int toss = 0; toss < number_of_tosses; toss++) {
            x = 2.0 * my_drand(&seed) - 1.0;
            y = 2.0 * my_drand(&seed) - 1.0;

            distance_squared = x * x + y * y;
            if (distance_squared <= 1) {
                local_count++;
            }
        }

        number_in_circle += local_count;
    }

    pi_estimate = 4.0 * number_in_circle / ((double)number_of_tosses);
    printf("Estimated value of pi: %f\n", pi_estimate);

    return 0;
}
