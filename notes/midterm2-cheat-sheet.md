### Performance Metrics
- **Response Time** The time taking to complete one task
- **Throughput** is the number of task completed per unit of time
- **CPU Time breakdown**
    - User time: Time the CPU spends running the users code
    - System time: Time CPU spent running the OS's code
    - Wait time: Time spent waiting for I/O or other services

---

### Instruction-Level Metrics

- **IPS (Instructions Per Second):** Approximate speed of CPU execution.
- **CPI (Cycles Per Instruction):**
  ```
  CPU Time = (CPI × Instruction Count) / Clock Rate
  ```

---

### Overhead in Parallelism

- **Overhead includes:**
  - Thread creation/destruction
  - Synchronization
  - Communication
  - Waiting due to load imbalance or mutual exclusion

---

### Speedup and Efficiency
formulas:
```
Speedup = T_serial / T_parallel
Efficiency = E = S / p // p is the number 
```
As p increases then E decreases due to overhead. If the problem size increases then both Speedup (S) and Efficiency (E) increase. due to less overhead

### Amdahl's Law
S is the max speedup, r is the percentage/fraction of the program that is parallelizable, p is the number of cores. r/p is also the parallel speedup
```
S = 1 / ((1 - r) + (r/p))
// as p approaches infinity
S = 1 / ((1 - r) + r)
```
that forumal is for fixed problem sizes
### Gustafson's law
This formula is for scalable/large problem sizes
If 'r' or the parallelizable portion is 100% then S = P 

**Strong scalability**  if E remains constant as p increases (that means the problem size is fixed). **Weak scalability** if E remains constant as both p and problem size increase.

---

## CUDA 
**Latency** is the time taken to complete one task. **Throughput** is the number of tasks completed per unit of time.

### CPU vs. GPU Architecture

| Feature | CPU | GPU |
|--------|-----|-----|
| Control logic | Complex | Simple |
| Threads | Few | Thousands |
| Memory bandwidth | Lower | Higher |
| Latency | Optimized | Higher (hidden) |
| Use case | Serial work | Parallel work |

GPU uses SIMD (single instruction multiple data). That's why GPU's are optimized for parallelism.

Host is the CPU, Device is the GPU, Kernel is the function run on the device (executed in parallel by many threads)
A grid is a collection of blocks, a block is a collection of threads

### Program Structure

- CUDA programs = **host code (CPU)** + **device code (GPU)**

```c
// Host
cudaMalloc();
cudaMemcpy();
// launch
kernel<<<blocks, threads>>>();
cudaMemcpy();
cudaFree();

// device
__global__ void kernel(...){ ... }

//launch the kernel
kernel<<<num_blocks, num_threads>>>(...);
```
---
### Function Qualifiers

| Qualifier | Runs on | Callable from |
|----------|---------|---------------|
| __global__ | Device | Host |
| __device__ | Device | Device |
| __host__ | Host | Host |
| __host__ __device__ | Both | Both |

---

```c
__global__ void add(int a, int b, int* c) {
    *c = a + b;
}

int main() {
    int *d_c;
    cudaMalloc(&d_c, sizeof(int));
    add<<<1,1>>>(2, 7, d_c);
    int c;
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2 + 7 = %d\n", c);
    cudaFree(d_c);
}
```