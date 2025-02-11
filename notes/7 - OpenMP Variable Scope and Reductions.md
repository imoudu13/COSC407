# OpenMP Variable Scope and Reductions

## Overview
OpenMP provides mechanisms for controlling **variable scope** and efficient **reduction operations** to aggregate data across threads.

### Key Topics:
- Variable Scope in OpenMP
- Shared, Private, and Firstprivate Variables
- Reduction Operations

---

## Variable Scope
In OpenMP, variable scope determines which **threads** can access a variable inside a parallel block.

### Shared Variables
- Exist in **one memory location**, accessible by all threads.
- Default behavior for variables declared **before** the parallel block.

```c
int x = 5;
#pragma omp parallel
{
    // All threads access the same x
}
```

### Private Variables
- Each thread gets **its own copy** of the variable.
- Uninitialized unless explicitly set.

```c
int y = 5;
#pragma omp parallel private(y)
{
    // Each thread gets its own y (uninitialized)
}
```

### Firstprivate Variables
- Like `private`, but **initialized** with the original value.

```c
int z = 5;
#pragma omp parallel firstprivate(z)
{
    // Each thread gets its own z, initialized to 5
}
```

### Default Clause
Sets the default scope for all variables.

```c
int x = 0, y = 0;
#pragma omp parallel num_threads(4) default(none) private(x) shared(y)
{
    x = omp_get_thread_num();
    #pragma omp atomic
    y += x;
}
```

---

## Reductions
**Reduction** operations allow threads to **aggregate results** safely without manual synchronization.

### Syntax
```c
#pragma omp parallel reduction(<operator> : <variable list>)
```

### Example 1: Summing Across Threads
```c
int sum = 0;
#pragma omp parallel reduction(+:sum)
{
    sum += omp_get_thread_num();
}
printf("Total sum = %d", sum);
```

### Example 2: Multiple Variables
```c
int x = 10, y = 10;
#pragma omp parallel reduction(+:x, y)
{
    x = omp_get_thread_num();
    y = 5;
}
printf("Shared: x=%d, y=%d\n", x, y);
```

### Reduction Operations
| Operator | Description |
|----------|-------------|
| `+` | Summation |
| `*` | Multiplication |
| `&` | Bitwise AND |
| `|` | Bitwise OR |
| `^` | Bitwise XOR |
| `&&` | Logical AND |
| `||` | Logical OR |

---

## Parallel Summation with Reduction
Instead of using a **critical section**, reductions optimize aggregation.

```c
double global_sum = 0;
#pragma omp parallel num_threads(4) reduction(+:global_sum)
{
    global_sum += compute_value(omp_get_thread_num());
}
```

---

## Area Under a Curve (Trapezoidal Rule)
Using **reduction** to integrate a function:

```c
double global_result = 0.0;
#pragma omp parallel num_threads(4) reduction(+:global_result)
{
    global_result += Local_trap(a, b, n);
}
printf("Approximate area: %f\n", global_result);
```

---

## Summary
- **Shared variables** exist in a single memory location.
- **Private variables** are unique per thread.
- **Firstprivate** ensures **initialized copies** for each thread.
- **Reduction** simplifies aggregating results across threads.

