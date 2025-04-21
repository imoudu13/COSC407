### OpenMP Programming Exam

#### Multiple Choice Questions

**Q1.** What does the `nowait` clause do in OpenMP?  
A) Skips to the next OpenMP construct  
B) Removes the synchronization barrier for the current construct  
C) Prioritizes the following OpenMP construct  
D) Removes the barrier from the previous construct  

**Q2.** In OpenMP, what is the collection of threads executing a parallel block called?  
A) Implicit task  
B) Parallel warp  
C) Team  
D) Executable code  

**Q3.** Which OpenMP directive creates a team of threads?  
A) `critical`  
B) `parallel`  
C) `single`  
D) `synchronization`  

**Q4.** Which directive ensures atomic memory updates?  
A) `sections`  
B) `for`  
C) `atomic`  
D) `parallel`  

**Q5.** Which code is correct for accumulating a sum?  
```c
// Code 1
#pragma omp atomic
x += tmp;

// Code 2 
#pragma omp critical
x += tmp;
```
A) Only Code 1  
B) Only Code 2  
C) Both, but Code 1 is faster  
D) Both are equally good

**Q6.** For non-atomic operations like `x = h()`, which is correct?
```c
// Code 1
#pragma omp atomic
x = h();

// Code 2
#pragma omp critical
x = h();
```
A) Code 1 is correct  
B) Code 2 is correct  
C) Both are correct  
D) Neither is correct

**Q7.** Which is correct for complex expressions?
```c
// Code 1
#pragma omp atomic
x *= 3*y + 2*x;

// Code 2
#pragma omp critical
x *= 3*y + 2*x;
```
A) Code 1 works  
B) Code 2 is required  
C) Both work  
D) Neither works

**Q8.** Does this code have a data race?
```c
omp_lock_t lock;
#pragma omp parallel {
    omp_set_lock(&lock);
    // variable updates
    omp_unset_lock(&lock);
}
```
A) Yes  
B) No

**Q9.** Output of balanced increment/decrement with separate critical sections?
```c
#pragma omp critical(a) x++;  // 50 threads
#pragma omp critical(b) x--;  // 50 threads
```
A) Always 0  
B) Sometimes non-zero  
C) Undefined

**Q10.** Output with same critical section names?
```c
#pragma omp critical x++; 
#pragma omp critical x--;
```
A) Always 0  
B) Sometimes non-zero  
C) Undefined

**Q11.** What happens with nested identical critical sections?
```c
#pragma omp critical {
    y += sqrt(1);  // sqrt() also has #pragma omp critical
}
```
A) Deadlock guaranteed  
B) Possible deadlock  
C) No deadlock

**Q12.** What if nested critical sections have different names?
```c
#pragma omp critical(A) {
    y += sqrt(1);  // sqrt() uses #pragma omp critical(B)
}
```
A) Deadlock guaranteed  
B) Possible deadlock  
C) No deadlock

**Q13.** Which header is required for OpenMP?  
A) `<omp.h>`  
B) `<parallel.h>`  
C) `<threads.h>`

**Q14.** What is the default number of threads created?  
A) Equal to CPU cores  
B) Environment variable determines  
C) Always 4

---

### Solutions

**Q1:** B) Removes the synchronization barrier for the current construct
- *Eliminates implicit barrier at end of worksharing constructs*

**Q2:** C) Team
- *Master thread + worker threads form a "team"*

**Q3:** B) `parallel`
- *Spawns a team of threads*

**Q4:** C) `atomic`
- *Provides low-overhead atomic operations*

**Q5:** C) Both, but Code 1 is faster
- *atomic has less overhead for simple operations*

**Q6:** B) Code 2 is correct
- *atomic only supports simple operations*

**Q7:** B) Code 2 is required
- *critical handles complex expressions*

**Q8:** B) No
- *Locks prevent concurrent access*

**Q9:** A) Always 0
- *Balanced operations with synchronization*

**Q10:** A) Always 0
- *Critical sections serialize access*

**Q11:** A) Deadlock guaranteed
- *Thread waits for itself in nested critical*

**Q12:** C) No deadlock
- *Different names allow nesting*

**Q13:** A) `<omp.h>`
- *Required for OpenMP functions*

**Q14:** B) Environment variable determines
- *OMP_NUM_THREADS controls default*

---

### Key Concepts
1. **Synchronization**
    - `nowait`: Eliminates implicit barriers
    - `atomic`: For simple memory operations
    - `critical`: For complex operations

2. **Thread Teams**
    - Created via `parallel` directive
    - Master + workers = team

3. **Deadlock Prevention**
    - Avoid identical nested critical sections
    - Use locks for fine-grained control

4. **Performance**
    - Prefer `atomic` over `critical` when possible
    - Minimize synchronization scope