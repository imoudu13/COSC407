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

```
S = T_serial / T_parallel
```