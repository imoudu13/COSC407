```c
pthread_mutex_t lock;
void* say_something(void *ptr) {
    pthread_mutex_lock(&lock);
    printf("%s ", (char*)ptr);
    pthread_mutex_unlock(&lock);
    pthread_exit(0);
}
int main() {
    pthread_t t1, t2;
    char *msg1 = "Hello ", *msg2 = "World!";
    pthread_mutex_init(&lock, NULL);
    pthread_create(&t1, NULL, say_something, msg1);
    pthread_create(&t2, NULL, say_something, msg2);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("Done!");
    pthread_mutex_destroy(&lock);
    exit(0);
}
```
**Mutex synchronization for thread safety.**

```c
#pragma omp parallel num_threads(4)
{ 
  int id = omp_get_thread_num();
  printf("T%d:A\nT%d:B\n", id, id);
  if (id == 0) printf("T0:special task\n");
  if (id == 1) printf("T1:special task\n");
  if (id == 2) printf("T2:special task\n");
}
printf("End");
```
**Task parallelism in OpenMP.**

```c
#pragma omp parallel num_threads(2)
{
  int id = omp_get_thread_num();
  int my_a = id * 3, my_b = id * 3 + 3;
  printf("T%d will process indexes %d to %d\n", id, my_a, my_b);
  for (int index = my_a; index < my_b; index++) printf("do work\n");
}
printf("done\n");
```
**Data parallelism using OpenMP.**

```c
#pragma omp parallel
{
    global_sum += my_sum;
}
```
**Race condition due to unsynchronized access.**

```c
#pragma omp atomic
sum += value;
```
**Atomic directive ensures safe updates.**

```c
#pragma omp critical
{
    shared_var += local_val;
}
```
**Critical section to prevent concurrent access.**

```c
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++)
    sum += array[i];
```
**Reduction safely aggregates results.**

```c
#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        C[i][j] = 0;
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
    }
```
**Parallel matrix multiplication using OpenMP.**

```c
#pragma omp parallel for schedule(dynamic,2)
for (int i = 0; i<8; i++)
    printf("T%d: %d\n", omp_get_thread_num(), i);
```
**Dynamic scheduling distributes workload efficiently.**
