# Parallel Concepts

## Concurrency
- When multiple operations are making progress within the same time period.
- Usually on the same core/thread.

## Parallelism
- When multiple operations are making progress at the same time.
- Usually requires multiple threads/cores.

## Process
An instance of the computer that is being executed. These are its components:
- Executable machine language program.
- Block of memory.
- Descriptor of the OS resources allocated to it.
- Security info.
- Information about the state of the process.

## Threading
- Threads are contained within processes.
- They allow programmers to divide their programs into independent tasks.
- A stream of instructions that can be scheduled to run independently from its main program.
- The hope is that when one thread blocks because it is waiting for resources, the other can run.

## Processes vs Threads
- Threads exist within a process; they’re like the children of the process.
- A process has at least one thread.
- If a process has more than one thread, it is multithreaded.
- Starting a thread within a process is known as **forking**.
- Terminating a thread is known as **joining**.
- Both threads and processes are units of execution or **tasks**.
- **Processes do not share memory** (each gets its own block of memory from the system).
- **Threads within a process share memory** (since they are children of the process, they have access to its resources).
- Data stored in a process's memory can be **shared or private**:
    - If **private**, only the thread that owns it can use it.

## How is Data Shared?
### Shared Memory
- Allows processors to have access to a global address space.
- Multiple processes can operate independently but share the same memory resources.
- Changes in a memory location affected by one task are visible to others.

### Uniform Memory Access (UMA)
- The time to access all memory locations is the same for all cores.

### Non-Uniform Memory Access (NUMA)
- A memory location a core is directly connected to can be accessed faster than a memory location that must be accessed through another chip.

## Task Scheduling
- A **scheduler** is a program that uses a **scheduling policy** to decide which process should run next.
- It uses a **selection function** to make the decision.
- The selection function considers:
    - Resources the process requires.
    - Time the process has been waiting.
    - The process's priority.
- Scheduling policy should try to optimize:
    - **Responsiveness** of interactive processes.
    - **Turnaround time** (the time the user waits for the process to finish).
    - **Resource utilization**.
    - **Fairness** (ensuring each process gets a chance to run).

## Some Scheduling Policies
### Non-Preemptive Policies
- Each task runs to completion before the next one can run.
- **First In First Out (FIFO)**.
- **Shortest-Job-First (SJF)**.

### Preemptive Policies
- **Round-Robin**: Each task is assigned a fixed time before it is required to give way to the next task and move back to the queue.
- **Earliest-Deadline-First**: The process with the closest deadline is picked next.
- **Shortest Remaining Time First**: The process with the shortest remaining time is picked first.

## Key Terms
- **Shared resource**: A resource available to all processes in the concurrent program.
- **Critical section**: Sections of code within a process that require access to shared resources. Cannot be executed while another process is in a corresponding section of code.
- **Mutual exclusion**: Requirement that when one process is in a critical section accessing a shared resource, no other process may be in a critical section accessing any of those shared resources.
- **Condition synchronization**: A mechanism ensuring that a process does not proceed until a certain condition is satisfied.
- **Deadlock**: A situation where two or more processes are unable to proceed because each is waiting for another process to act.
- **Livelock**: A situation where two or more processes continuously change their state in response to changes in other processes without making any progress.
- **Race Condition**: A situation where multiple tasks read/write a shared data item, and the result depends on the relative timing of their execution.
- **Starvation**: A situation where a runnable process is overlooked indefinitely by the scheduler.

## Dead or Alive(lock)
Concurrent programs must satisfy two properties:
1. **Safety**: The program doesn’t enter a bad state.
2. **Liveness**: The program must progress.

Two problems that can occur:
- **Deadlock**: A process is waiting for a shared resource that will never be available (e.g., another process is waiting for this process to act).
- **Livelock**: Multiple processes continuously change state in response to each other without making progress.

### Conditions for Deadlock
For deadlock to occur, **four conditions must hold**:
1. **Mutual Exclusion**: The program involves a shared resource protected by mutual exclusion.
2. **Hold While Waiting**: A process can hold a resource while waiting for others.
3. **No Preemption**: The OS cannot force a process to deallocate a resource it holds.
4. **Circular Wait**: P1 is waiting for a resource held by P2, and P2 is waiting for a resource held by P1.

### Preventing Deadlock
To prevent deadlock, **prevent at least one of the four conditions** from occurring.

# POSIX Threads
A **POSIX thread** is a thread associated with a process’s shared resources. Each thread has its own:
- **Stack**
- **Program counter**
- **Registers**
- **Thread ID**

## Races
A **race condition** occurs when the **parent process exits before its child threads complete**. This does not allow enough time for child threads to finish execution.

## Fixes for Race Conditions
- Best fix for race conditions, use mutual exclusions and join the threads

```
pthread_mutex_t lock;

void* say_something(void *ptr) {
    pthread_mutex_lock(&lock); //this now becomes critical section! it uses mutual exclusion
    printf("%s ", (char*)ptr);
    pthread_mutex_unlock(&lock); //end the critical condition
    pthread_exit(0);
}

int main() {
    pthread_t thread_1, thread_2;
    char *msg1 = "Hello ";
    char *msg2 = "World!";
    //  create the lock -> error checking?
    pthread_mutex_init(&lock, NULL);
    pthread_create( &thread_1, NULL, say_something, msg1);
    pthread_create( &thread_2, NULL, say_something, msg2);

    // the main thread has to wait for the other threads to terminate before it can terminate
    pthread_join(thread_1, NULL);
    pthread_join(thread_2, NULL);

    printf("Done!");
    fflush(stdout);
    pthread_mutex_destroy(&lock);
    exit(0);
}

```

This is conditional synchronization
```
void* say_something(void *ptr) {
    pthread_mutex_lock(&lock);//this now becomes critical section!
    //check on some condition - if it is hello, wait for world....
    
    if (strcmp(”World!",(char*)ptr) == 0) {
        printf("Waiting on condition variable cond1\n");
        if (done == 0) //only wait in the event that you need to…
            pthread_cond_wait(&cond1, &lock);
    } else {
        printf("Signaling condition variable cond1\n");
        done == 1;
        pthread_cond_signal(&cond1);
    }
    
    printf("%s ", (char*)ptr);
    pthread_mutex_unlock(&lock);
    pthread_exit(0);
}
```

# 5 - Intro to OpenMP
OpenMP = open Multi-Processing

An api for multithreaded shared parallel programming

OpenMP is: 
- higher level than Pthreads
- programmer only states that a block of code is to be executed in parallel
- requires compiler support


### Task Parallelisms
- Share the tasks among each core ie on core does the tasks on all data
### Data parallelism
- Share the data among each core

## OpenMP API
OpenMP is based on directives

### OpenMP API components
- Compiler directive
- Runtime library routines
- Environment variables

## Fork Join
OpenMP uses the fork join model. The enforces synchronization so every thread must wait till everyone is finished <br>
before proceeding to the next region
A group of threads executing the parallel block is known as a team, the original thread is called the master, <br>
the children are called slaves

### Task parallelism
```c
#pragma omp parallel num_threads(4)
{ 
  int id = omp_get_thread_num();
  printf("T%d:A\n", id);
  printf("T%d:B\n", id);
  
  if (id == 0)
    printf("T0:special task\n");
    
  if (id == 1)
    printf("T1:special task\n");
    
  if(id == 2)
    printf("T2:special task\n");
}

printf("End");
```

### Data Parallelism
```C
#pragma omp parallel num_threads(2)
{
  int id = get_thread_num()
  int my_a = id * 3;  \\ where you want the thread to start doing work
  int my_b = id * 3 + 3; \\ where it should stop doing work
  
  printf("T%d will process indexes %d to ");
  
  for (int index = my_a; index < my_b; index++) 
    printf("do work\n");
}

printf("done\n");

return 0;
```

# 6 - OpenMP Mutexes, Exclusions, and Synchronization

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

