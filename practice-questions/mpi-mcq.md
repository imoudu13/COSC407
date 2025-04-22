### MPI Practice Questions

#### Multiple Choice Questions

**Q1.**  
What is the output? Assume comm_sz = 3.  
```c
if (my_rank != 0)
    MPI_Send(&my_rank, 1, MPI_INT, 0, ...);
else {
    printf("%d ", my_rank);
    MPI_Recv(&x, 1, MPI_INT, 1, ...);
    printf("%d ", x);
}
```
A. 01  
B. 123  
C. 012  
D. Same as (B) but in any order (e.g., 2,1,0)  
E. 0 then print 1, 2 in any order

---

**Q2.**  
What is the output? Assume comm_sz = 3.
```c
if (my_rank != 0)
    MPI_Send(&my_rank, 1, MPI_INT, 0, ...);
else {
    printf("%d ", my_rank);
    MPI_Recv(&x, 1, MPI_INT, MPI_ANY_SOURCE, ...);
    printf("%d ", x);
}
```
A. 01 or 02  
B. 123  
C. 012  
D. Same as (B) but in any order (e.g., 2,1,0)  
E. 0 then print 1, 2 in any order

---

**Q3.**  
What is the output? Assume comm_sz = 4.
```c
if (my_rank != 0)
    MPI_Send(&my_rank, 1, MPI_INT, 0, ...);
else {
    printf("%d ", my_rank);
    for (int q = 1; q < comm_sz; q++) {
        MPI_Recv(&x, 1, MPI_INT, MPI_ANY_SOURCE, ...);
        printf("%d ", x);
    }
}
```
A. 01 or 02  
B. 123  
C. 012  
D. Same as (B) but in any order (e.g., 2,1,0)  
E. 0 then print 1, 2, 3 in any order

---

#### Programming Questions

**Q4 (Vector Average):**  
Write an MPI program where:
- Process 0 reads a vector
- All processes compute the average
- Process 0 prints the average  
  *Assume vector size `N` is divisible by `comm_sz`.*

---

**Q5 (Ordered Output):**  
Rewrite the following code to print outputs in rank order (P0 first, P1 next, etc.):
```c
printf("P %d of %d: Hi\n", my_rank, comm_sz);
```

---

**Q6 (Message Pipeline):**  
Write a program where:
- Each non-zero process sends 10 tagged messages to Process 0
- Process 0 receives all messages with `MPI_ANY_TAG` and prints them ordered by:  
  a) Sender rank  
  b) Tag

---

**Q7 (Prime Counting):**  
Parallelize the prime-counting algorithm using:
1. **Block distribution**
2. **Cyclic distribution**  
   *Explain why cyclic is better for load balancing.*

---

### Solutions

**Q1:** A. 01
- *Explanation:* Rank 0 prints `0`, then only receives from rank 1 (ignores rank 2).

**Q2:** A. 01 or 02
- *Explanation:* Rank 0 prints `0`, then receives from *either* rank 1 or 2 (due to `MPI_ANY_SOURCE`).

**Q3:** E. 0 then print 1, 2, 3 in any order
- *Explanation:* Rank 0 prints `0` first, then receives from all other ranks in arbitrary order.

---

**Q4 (Vector Average):**
```c
MPI_Scatter(A, my_n, MPI_FLOAT, my_A, my_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
float my_sum = 0;
for (int i = 0; i < my_n; i++) my_sum += my_A[i];
MPI_Reduce(&my_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
if (my_rank == 0) printf("Avg: %f\n", global_sum / N);
```

**Q5 (Ordered Output):**
```c
if (my_rank == 0) {
    printf("P %d of %d: Hi\n", my_rank, comm_sz);
    for (int src = 1; src < comm_sz; src++) {
        MPI_Recv(msg, MAX, MPI_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%s", msg);
    }
} else {
    sprintf(msg, "P %d of %d: Hi\n", my_rank, comm_sz);
    MPI_Send(msg, strlen(msg)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}
```

**Q6 (Message Pipeline):**
```c
// Sender (non-zero ranks):
for (int tag = 0; tag < 10; tag++) {
    MPI_Send(&data, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
}
// Receiver (rank 0):
for (int prc = 1; prc < comm_sz; prc++) {
    for (int tag = 0; tag < 10; tag++) {
        MPI_Recv(&data, 1, MPI_INT, prc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%d from P%d, tag %d\n", data, prc, tag);
    }
}
```

**Q7 (Prime Counting):**
- **Block Distribution:**
  ```c
  for (int i = my_a; i < my_b; i++) { /* prime check */ }
  ```
  *Disadvantage:* Higher ranks get larger numbers (more computation).

- **Cyclic Distribution:**
  ```c
  for (int i = 2 + my_rank; i <= n; i += comm_sz) { /* prime check */ }
  ```
  *Advantage:* Evenly distributes large/small numbers across processes.
```

### Key Notes:
1. **Collective Communication**: All processes must call collective functions (e.g., `MPI_Scatter`) to avoid deadlocks.
2. **Load Balancing**: Cyclic distribution outperforms block distribution for irregular workloads (e.g., prime checking).
3. **Ordered Output**: Use rank 0 as a coordinator to sequence outputs from other processes.