### PART A: Multiple Choice Questions

**Q1.**  
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
a) 0 1 or 0 2  
b) 1 2 3  
c) 0 1 2  
d) Same as (B) but in any order (e.g., 2,1,0)  
e) 0 then print 1, 2 in any order

**Q2.**  
Consider the following code fragment:
```c
int n;  
scanf("%d", &n);  
print("%d", n);  
int a = 2;  
int b = 3;  
print("%d", a);
```
What is the maximum theoretical speedup?  
A. 1  
B. 2  
C. 4  
D. 8

**Q3.**  
Consider the following code fragment:
```c
int n;  
scanf("%d", &n);  
print("%d", n);  
int a = n;  
int b = 3;
```
What is the maximum theoretical speedup?  
A. 1.0  
B. 1.5  
C. 1.7  
D. 2.0

**Q4.**  
A program has \( r = 90\% \) (10% cannot be parallelized). According to Amdahl’s Law, what is the maximum speedup?  
A. 5  
B. 10  
C. 2  
D. 100

**Q5.**  
Consider the following code:
```c
int sum = 0, tid;
#pragma omp parallel num_threads(32)
{
    tid = omp_get_thread_num();
    #pragma omp atomic
    sum += tid;
}
printf("sum= %d\n", sum);
```
Select the correct statement:  
A. Compile time error  
B. No error and correct output all the time  
C. Runtime error  
D. No error but wrong output sometimes

### PART B: Short and Long Answer Questions

**Short Answer Q1:**  
Two of the most famous laws in parallel computing are Amdahl’s Law and Gustafson’s Law. Explain which of these is more relevant to GPU programming and why.

**Long Answer Q1:**  
Write an MPI program where:
- Process 0 reads a vector
- All processes work together to find the average
- Process 0 prints the average  
  Assumptions:
- Vector size N is divisible by comm_sz
- `readVector()` function exists (no need to implement).

**Long Answer Q2:**  
Complete the CUDA implementation for summing an array using shared memory reduction. Address:
1. GPU memory allocation
2. Data copy to GPU
3. Grid size calculation
4. Kernel `gpu_sum` for partial block sums
5. Kernel launch
6. Copy partial sums to host
7. Final host summation + error checking

**Long Answer Q3:**  
Parallelize the trapezoidal rule calculation with OpenMP **without** using `reduction`.
- Manually divide trapezoids among threads
- Use thread-safe accumulation (e.g., critical/atomic)
- Combine partial results explicitly

**Long Answer Q4:**  
Complete the CUDA implementation for squaring array elements. Address:
1. GPU memory allocation
2. Data copy to GPU
3. Grid size calculation
4. Kernel `gpu_sqr`
5. Kernel launch
6. Result copy to host
7. Error checking

---

### Solutions

**Q1:** a) 0 1 or 0 2
- *Explanation:* Rank 0 prints its ID first, then receives either 1 or 2 from other ranks.

**Q2:** A. 1
- *Explanation:* The serial region (3 instructions) dominates; \( r = 0.5 \), speedup = 1.

**Q3:** A. 1.0
- *Explanation:* Data dependency on `n` makes the entire code serial (\( r = 0 \)).

**Q4:** B. 10
- *Explanation:* Amdahl’s Law: \( \text{Speedup} = 1 / (1 - 0.9) = 10 \).

**Q5:** B. No error and correct output all the time
- *Explanation:* `atomic` ensures correct summation; output is always 496 (sum of 0..31).

**Short Answer Q1:**
- *Gustafson’s Law* is more relevant to GPUs because it assumes problem size scales with resources (fitting GPU’s massively parallel architecture), whereas Amdahl’s Law focuses on fixed problem sizes.