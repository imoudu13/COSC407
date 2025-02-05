# OpenMP Mutexes, Exclusions, and Synchronization

## Overview
OpenMP provides mechanisms for **synchronization and mutual exclusion** to prevent **race conditions** and ensure **safe parallel execution**.

### Key Topics:
- Race Conditions
- Barriers
- Mutual Exclusion (Critical Sections, Atomics, and Locks)
- Performance Considerations

---

## Race Conditions
A **race condition** occurs when multiple threads **simultaneously access and modify shared data**, leading to **unpredictable behavior**.

### Example:
```c
#pragma omp parallel
{
    global_sum += my_sum; // Potential race condition
}
```
To prevent this, we use **mutual exclusion** techniques.

---

## Barriers
**Barriers** ensure that all threads reach a synchronization point before continuing execution.

### Types of Barriers:
1. **Implicit Barriers** - Automatically added at the end of parallel regions.
2. **Explicit Barriers** - Defined using `#pragma omp barrier`.

### Example:
```c
#pragma omp parallel
{
    compute_part();
    #pragma omp barrier // Ensures all threads finish before proceeding
    finalize_part();
}
```

### Barrier Limitations:
- All threads must encounter the barrier.
- Conditional execution may lead to **illegal barriers**.

---

## `nowait` Clause
Using `nowait` allows threads **to skip synchronization** when it is unnecessary, improving performance.

### Example:
```c
#pragma omp single nowait
{
    expensive_task();
}
// Other threads continue execution without waiting.
```

---

## Mutual Exclusion
**Mutual exclusion** ensures that only **one thread at a time** accesses a critical section.

### OpenMP Mutual Exclusion Mechanisms:
1. **Critical Directive** - Ensures exclusive execution.
2. **Atomic Directive** - Ensures atomic updates to a shared variable.
3. **Locks** - Explicit locking mechanisms.

### 1. Critical Directive
```c
#pragma omp critical
{
    shared_var += local_val;
}
```
**Named Critical Sections:**
```c
#pragma omp critical(name1)
x = compute_x();
#pragma omp critical(name2)
y = compute_y();
```
- Allows **simultaneous execution** of **different** critical sections.

---

### 2. Atomic Directive
`#pragma omp atomic` is **faster** than `critical` for **simple updates**.

```c
#pragma omp atomic
sum += value;
```

Supported Operations:
- `x++`, `x--`, `x += expr`, `x = x + expr`

---

### 3. Locks
Locks **manually enforce** mutual exclusion.

```c
#include <omp.h>
static omp_lock_t mylock;

int main() {
    omp_init_lock(&mylock);

    #pragma omp parallel
    {
        omp_set_lock(&mylock);
        critical_section();
        omp_unset_lock(&mylock);
    }

    omp_destroy_lock(&mylock);
    return 0;
}
```
**Key Lock Functions:**
- `omp_init_lock(&lock);`
- `omp_set_lock(&lock);`
- `omp_unset_lock(&lock);`
- `omp_destroy_lock(&lock);`

---

## When to Use Which?
| Mechanism  | Use Case |
|------------|----------|
| **Atomic** | Single-variable updates (fastest) |
| **Critical** | Protects complex code sections |
| **Locks** | Fine-grained control over execution |

---

## Caveats & Best Practices
1. **Avoid Mixing** different mutual exclusion methods.
2. **Fairness is NOT guaranteed** - Some threads may starve.
3. **Avoid Nesting** critical sections (deadlocks possible).

---

## Conclusion
- **Barriers ensure synchronization** among threads.
- **Critical sections and atomics prevent race conditions**.
- **Locks provide explicit control** over execution.
- **Performance trade-offs exist between mechanisms**.

### Next Steps:
- Learn about **variable scopes (shared, private, firstprivate)**.
- Explore **Reduction and Work Sharing**.

**Further Reading:**
- [OpenMP Documentation](https://www.openmp.org/specifications)
- OpenMP Resources in Course Materials

