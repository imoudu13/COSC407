# POSIX Threads (Pthreads)

## Introduction
POSIX Threads (Pthreads) provide a standard interface for multithreading in C. Threads within a process share resources but maintain their own execution context.

### Key Topics:
- Parallel concepts with `pthreads`
- Thread management
- Synchronization
- Mutexes
- Condition variables

---

## Thread Characteristics
Threads associated with a process share:
- **Heap memory**
- **Global variables**
- **File descriptors**

Each thread has its own:
- **Stack (private variables)**
- **Program Counter (PC)**
- **Registers**
- **Thread ID**

---

## Creating Threads
Threads are created using `pthread_create()`, which takes four arguments:
1. **Thread variable** - Holds the reference to the thread
2. **Thread attribute** - Specifies properties (e.g., stack size)
3. **Function pointer** - The function the thread will execute
4. **Arguments** - Arguments to pass to the function

### Example:
```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* thread_function(void *argument) {
    printf("%s\n", (char*)argument);
    return NULL;
}

int main() {
    pthread_t thread;
    char *message = "Hello from thread";

    pthread_create(&thread, NULL, thread_function, message);
    pthread_join(thread, NULL); // Wait for the thread to finish

    printf("Main thread done\n");
    return 0;
}
```

---

## Race Conditions
A race condition occurs when multiple threads access shared resources concurrently, leading to unpredictable behavior.

### Example:
```c
#include <stdio.h>
#include <pthread.h>

void* say_something(void *ptr) {
    printf("%s ", (char*)ptr);
    return NULL;
}

int main() {
    pthread_t thread1, thread2;

    char *msg1 = "Hello";
    char *msg2 = "World";

    pthread_create(&thread1, NULL, say_something, msg1);
    pthread_create(&thread2, NULL, say_something, msg2);

    printf("Done!\n");
    return 0;
}
```
**Potential Output Issues:**
- Output order may be inconsistent (`Hello World`, `World Hello`, etc.)
- Parent thread may exit before child threads finish

---

## Synchronization Using `pthread_join()`
The `pthread_join()` function ensures a thread completes before the parent thread proceeds.

### Example:
```c
pthread_join(thread1, NULL);
pthread_join(thread2, NULL);
```

---

## Mutex (Mutual Exclusion)
A **mutex** is used to ensure only one thread accesses a critical section at a time.

### Example:
```c
#include <stdio.h>
#include <pthread.h>

pthread_mutex_t lock;

void* safe_print(void *ptr) {
    pthread_mutex_lock(&lock);
    printf("%s\n", (char*)ptr);
    pthread_mutex_unlock(&lock);
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    char *msg1 = "Hello";
    char *msg2 = "World";

    pthread_mutex_init(&lock, NULL);

    pthread_create(&thread1, NULL, safe_print, msg1);
    pthread_create(&thread2, NULL, safe_print, msg2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&lock);
    return 0;
}
```
**Key Mutex Functions:**
- `pthread_mutex_init(&lock, NULL);` → Initialize mutex
- `pthread_mutex_lock(&lock);` → Lock the critical section
- `pthread_mutex_unlock(&lock);` → Unlock the critical section
- `pthread_mutex_destroy(&lock);` → Destroy mutex after use

---

## Condition Variables
Condition variables allow threads to wait for a specific condition to be met before continuing execution.

### Example:
```c
#include <stdio.h>
#include <pthread.h>
#include <string.h>

pthread_mutex_t lock;
pthread_cond_t cond;
int done = 0;

void* say_something(void *ptr) {
    pthread_mutex_lock(&lock);
    if (strcmp("World!", (char*)ptr) == 0) {
        if (done == 0)
            pthread_cond_wait(&cond, &lock);
    } else {
        done = 1;
        pthread_cond_signal(&cond);
    }
    printf("%s ", (char*)ptr);
    pthread_mutex_unlock(&lock);
    return NULL;
}

int main() {
    pthread_t thread1, thread2;
    char *msg1 = "Hello";
    char *msg2 = "World!";

    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&cond, NULL);

    pthread_create(&thread1, NULL, say_something, msg1);
    pthread_create(&thread2, NULL, say_something, msg2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&cond);
    return 0;
}
```
**Key Condition Variable Functions:**
- `pthread_cond_wait(&cond, &lock);` → Wait for condition
- `pthread_cond_signal(&cond);` → Signal waiting threads

---

## Summary of Key Functions
| Function | Description |
|----------|-------------|
| `pthread_create()` | Creates a thread |
| `pthread_join()` | Waits for a thread to complete |
| `pthread_mutex_init()` | Initializes a mutex |
| `pthread_mutex_lock()` | Locks a mutex |
| `pthread_mutex_unlock()` | Unlocks a mutex |
| `pthread_cond_wait()` | Waits on a condition variable |
| `pthread_cond_signal()` | Signals a condition variable |

---

## Conclusion
- **POSIX Threads provide multithreading capabilities in C.**
- **Synchronization is crucial to prevent race conditions.**
- **Mutexes and condition variables help manage thread execution.**
- **`pthread_join()` ensures proper thread completion.**

### Next Steps:
- Explore **OpenMP** for high-level parallel programming.
- Check **[POSIX Threads Programming](https://hpc-tutorials.llnl.gov/posix/)** for in-depth tutorials.

