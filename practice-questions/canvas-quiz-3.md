### CUDA Programming Exam

#### Multiple Choice Questions

**Q1.** Data parallelism refers to:  
A) Serial execution of different instructions  
B) Concurrent execution of same instructions on different data <br>
C) Different instructions on single core  

**Q2.** GPU parallelization is always better than CPU:  
A) True  
B) False

**Q3.** CPUs optimize ____ while GPUs optimize ____:  
A) Latency, throughput  
B) Throughput, latency  
C) Both optimize latency  

**Q4.** Amdahl's Law: 30% parallelizable → max speedup:  
A) 1.25×  
B) 1.43×<br>
C) 2.00×  

**Q5.** 99% parallelized with 100× speedup → overall:  
A) 50×  
B) 75×  
C) 100×  

**Q6.** 99.9% parallelized with 1000× speedup → overall:  
A) 100×  
B) 500×  
C) 1000×  

**Q7.** Invalid block config for 100×100 image:  
A) Many 1D blocks  
B) One 100×100 block  
C) Many 2D blocks  

**Q8.** Least preferred for 32×32 image:  
A) Many 1D blocks  
B) One 32×32 block <br>
C) Many 2D blocks  

**Q9.** Asterisks printed: `foo<<<3, dim3(2,2)>>>()`:  
A) 3  
B) 4  
C) 12  

**Q10.** `dim3 grid(5,10,2); dim3 block(4,4);` → total:  
A) 100 blocks, 16 threads  
B) 100 blocks, 1600 threads 
C) 17 blocks, 136 threads  

**Q11.** Launch 1M threads in 1 block for 1000×1000 matrix:  
A) True  
B) False   

**Q12.** 1 block with 1024×1024 threads:  
A) True  
B) False 

**Q13.** 1024 blocks × 1024 threads/block:  
A) Valid
B) Invalid  

**Q14.** Matrix storage in CUDA:  
A) 2D grid  
B) Column-major 1D  
C) Row-major 1D  

**Q15.** `<<<>>>` syntax:  
A) Launches kernel 
B) Allocates memory  
C) Copies data  

**Q16.** `A[A[i]]=7` access pattern:  
A) Coalesced read, random write 
B) Strided read/write  
C) Fully coalesced  

**Q17.** Warp-aligned vs non-aligned divergence:  
A) Non-aligned better  
B) Warp-aligned better 
C) No difference  

**Q18.** Access patterns:  
```c
x = A[i];         // (1) 
x = A[i+16];      // (2)
x = A[128-i];     // (3)
```
A) All coalesced 
B) All strided  
C) Mixed

**Q19.** Memory speed (fastest to slowest):  
A) Registers > shared > local > global 
B) Shared > registers > global > local  
C) All equal

**Q20.** Operation speed ranking:
```c
y = x;            // (1)
arr[0] = arr[1];  // (2) 
sh2 = sh1;        // (3)
z = arr[10];      // (4)
```
A) 1 > 3 > 4 > 2
B) 3 > 1 > 2 > 4  
C) All equal

**Q21.** Better matrix multiply:  
A) Accumulate in register then write
B) Write directly each iteration  
C) Equal

**Q22.** Where are M,N,P matrices stored?  
A) Global memory
B) Shared memory  
C) Registers

**Q23.** Where is `float value` stored?  
A) Global memory  
B) Register
C) Shared memory

**Q24.** Access patterns:
```c
x = A[2*i];       // (1)
x = A[2*i+1];     // (2) 
A[i] = x;         // (3)
```
A) (1),(2) strided; (3) coalesced
B) All strided  
C) All coalesced

**Q25.** Shared memory for matrix addition:  
A) Always helpful  
B) No benefit
C) Helps slightly

**Q26.** Tiled 32×32 matrix multiply bandwidth reduction:  
A) 1/8  
B) 1/16  
C) 1/32

**Q27.** Naive matrix multiply memory accesses per element:  
A) 1  
B) N
C) logN

**Q28.** Tiled (tile size T) matrix multiply accesses:  
A) N  
B) N/T
C) T

---

### Solutions

**Q1:** B - Same operation on different data segments  
**Q2:** B - Some workloads better on CPU  
**Q3:** A - CPUs low latency, GPUs high throughput  
**Q4:** B - 1/(1-0.3) ≈ 1.43  
**Q5:** A - 1/(0.01+0.0099) ≈ 50  
**Q6:** B - 1/(0.001+0.000999) ≈ 500  
**Q7:** B - Exceeds 1024 thread limit  
**Q8:** B - Underutilizes SMs  
**Q9:** C - 3 blocks × 4 threads = 12  
**Q10:** B - 100×16=1600  
**Q11:** B - Max 1024 threads/block  
**Q12:** B - 1M >> 1024 limit  
**Q13:** A - Valid if resources allow  
**Q14:** C - Row-major linear layout  
**Q15:** A - Kernel launch only  
**Q16:** A - Random write from unknown index  
**Q17:** B - Warp alignment reduces divergence  
**Q18:** A - All consecutive accesses  
**Q19:** A - Registers fastest, global slowest  
**Q20:** A - Register ops > shared > global  
**Q21:** A - Fewer global memory writes  
**Q22:** A - Default storage is global  
**Q23:** B - Local scalars go in registers  
**Q24:** A - Strided reads, coalesced write  
**Q25:** B - No data reuse in simple addition  
**Q26:** C - Each element loaded once per tile  
**Q27:** B - Each element used N times  
**Q28:** B - N/T tiles × 1 load

---

### Key Concepts
1. **Thread Hierarchy**
    - Blocks limited to 1024 threads
    - Grids can have many blocks
    - 2D/3D organization supported

2. **Memory Hierarchy**
    - Registers (fastest, thread-private)
    - Shared memory (block-level)
    - Global memory (slowest, persistent)

3. **Optimization**
    - Coalesced memory access
    - Minimize global memory operations
    - Use shared memory for data reuse

4. **Performance**
    - Occupancy vs. resource usage
    - Warp divergence impacts
    - Tiling reduces memory traffic
