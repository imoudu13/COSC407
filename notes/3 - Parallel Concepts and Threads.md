# Parallel Concepts and Threads

## 1. Sequential vs. Concurrent Systems

### **Sequential Systems**
- Each task must complete before the next starts.
- **Benefit**: Simple to design and use.
- **Drawback**: Can be slow since it waits on resources.

### **Concurrent Systems**
- Multiple activities run at the same time.
- **Types**:
  - **Parallel Systems**: Multiple processors execute tasks simultaneously.
  - **Pseudo-Parallel Systems**: Single processor time-shares tasks.

## 2. Concurrency vs. Parallelism
| Concept       | Definition |
|--------------|------------|
| **Concurrency** | Multiple operations make progress in the same period (interleaving) but not necessarily simultaneously. |
| **Parallelism** | Multiple operations execute at the exact same time on different processors. |

## 3. Processes and Threads

### **Processes**
- An instance of a program being executed.
- Contains executable code, memory, and OS resources.

### **Threads**
- A **lightweight** unit of execution inside a process.
- **Multithreading**: A process with multiple threads.
- **Benefits**:
  - More efficient than multiple processes.
  - Shared memory space.

## 4. Threading Models
### **POSIX Threads (Pthreads)**
- A standardized threading API for UNIX-like systems.
- **Why Pthreads?**
  - Lightweight (less overhead than processes).
  - Efficient data sharing (same address space).

### **Creating a Thread (C Example)**
```c
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void* say_hello(void* data) {
    char *str = (char*)data;
    while(1) {
        printf("%s\n", str);
        sleep(1);
    }
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, say_hello, "Hello from 1");
    pthread_create(&t2, NULL, say_hello, "Hello from 2");
    pthread_join(t1, NULL);
}
