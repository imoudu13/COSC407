# Introduction to OpenMP

## Overview
OpenMP (Open Multi-Processing) is an API for multi-threaded, shared-memory parallel programming. It is designed for systems where each thread or process has access to shared memory.

### Key Topics:
- Basics of OpenMP
- Parallelism using OpenMP
- Work distribution
- Synchronization

---

## OpenMP vs Pthreads

| Feature  | OpenMP | Pthreads |
|----------|--------|----------|
| **Abstraction Level** | High | Low |
| **Ease of Use** | Easier (compiler handles details) | Requires explicit thread management |
| **Compiler Support** | Requires a compiler with OpenMP support | Works with most C compilers |
| **Parallelism Control** | Uses directives (`#pragma omp`) | Uses thread functions |

---

## OpenMP API Components
OpenMP consists of three main components:
1. **Compiler directives** (`#pragma omp ...`)
2. **Runtime library routines** (`omp_get_thread_num()`)
3. **Environment variables** (e.g., `OMP_NUM_THREADS=4`)

---

## OpenMP Fork-Join Model
OpenMP follows a **fork-join** execution model:
- **Master thread** forks into multiple threads for parallel execution.
- **Threads execute tasks concurrently**.
- **Synchronization** ensures threads complete before proceeding.

---

## OpenMP Directives
### Basic Syntax:
```c
#pragma omp directive [clause]
```
### Example:
```c
#pragma omp parallel num_threads(4)
```

### Common Directives:
| Directive | Description |
|-----------|-------------|
| `#pragma omp parallel` | Creates a parallel region |
| `#pragma omp for` | Distributes loop iterations among threads |
| `#pragma omp sections` | Parallel execution of code blocks |
| `#pragma omp single` | Ensures only one thread executes the block |

---

## OpenMP Example: "Hello World"
### Serial Version:
```c
#include <stdio.h>
int main() {
    printf("Hello World!\n");
    return 0;
}
```
**Output:**
```sh
Hello World!
```

### Parallel Version:
```c
#include <stdio.h>
#include <omp.h>
int main() {
    #pragma omp parallel
    printf("Hello World!\n");
    return 0;
}
```
**Output:**
```sh
Hello World!
Hello World!
Hello World!
Hello World!
```

---

## Specifying Number of Threads
```c
#include <stdio.h>
#include <omp.h>
int main() {
    #pragma omp parallel num_threads(3)
    {
        int my_id = omp_get_thread_num();
        int tot = omp_get_num_threads();
        printf("Hello World from thread %d/%d\n", my_id, tot);
    }
    return 0;
}
```
**Output Example:**
```sh
Hello World from thread 2/3
Hello World from thread 0/3
Hello World from thread 1/3
```

---

## Assigning Tasks to Threads
```c
#pragma omp parallel num_threads(4)
{
    int id = omp_get_thread_num();
    printf("T%d: A\n", id);
    printf("T%d: B\n", id);
    if (id == 2) printf("T2: Special Task\n");
}
```

---

## Distributing Data Among Threads
```c
int list[6] = {0,1,2,3,4,5};
#pragma omp parallel num_threads(2)
{
    int id = omp_get_thread_num();
    int start = id * 3;
    int end = start + 3;
    for (int i = start; i < end; i++)
        printf("T%d: Processing list[%d]\n", id, i);
}
```

---

## Parallel Sum Calculation
### Serial Version:
```c
int sum = 0;
for (int i = 0; i < N; i++) {
    sum += array[i];
}
```
### Parallel Version:
```c
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++) {
    sum += array[i];
}
```

---

## Handling Compiler Incompatibility
```c
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
```

---

## Conclusion
- OpenMP simplifies parallel programming with minimal code modifications.
- It supports **parallel loops**, **task parallelism**, and **synchronization**.
- OpenMP is best suited for **multicore** shared-memory systems.

### Next Steps:
- Learn about **mutexes, atomic operations, and barriers**.
- Explore **OpenMP variable scopes** (shared, private, firstprivate).
- Study **reduction operations** for efficient parallel computation.

**Further Reading:**
- [OpenMP Specifications](http://www.openmp.org/specifications)
- OpenMP Resources (Check your course module)

