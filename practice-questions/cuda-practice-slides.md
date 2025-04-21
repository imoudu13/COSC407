### CUDA Programming Practice Exam

#### Problem A: Brute-force Search
**Task:** Implement brute-force search on an unsorted array of N elements (N ≤ 1 billion).

**Solution:**
```c
__device__ int index;  // Global variable to store found index

__global__ void brute_force_search(int key, int* a, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N && a[i] == key) {
        index = i;
        abort();  // Terminates all threads
    }
}
```

**Key Points:**
- Each thread checks one array element
- Uses global `index` variable and `abort()` for early termination
- Embarrassingly parallel problem

---

#### Problem B: Drawing Fractals (Von Koch Snowflake)
**Task:** Parallelize fractal drawing where each segment is independent.

**Solution:**
```c
__global__ void drawSnowFlake(int level) {
    float x0, y0, x1, y1;
    int N = pow(4, level-1);
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < N) {
        getXY(i, &x0, &y0, level);
        getXY(i+1, &x1, &y1, level);
        drawMotif(x0, y0, x1, y1);
    }
}
```

**Key Points:**
- No loop-carried dependencies
- Each thread handles one segment
- Perfectly parallel problem

---

#### Problem C: Count Sort
**Task:** Implement parallel count sort with optimizations.

**Optimized Solution:**
```c
__global__ void count_sort(int* a, int* a_sorted, int N) {
    __shared__ int a_sh[BLOCK_WIDTH];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    if (i < N) {
        int value = a[i];
        int count = 0;
        
        // Tiled approach
        for (int m = 0; m < (N + BLOCK_WIDTH - 1)/BLOCK_WIDTH; m++) {
            int load_idx = tid + m * BLOCK_WIDTH;
            if (load_idx < N) 
                a_sh[tid] = a[load_idx];
            __syncthreads();
            
            for (int j = 0; j < BLOCK_WIDTH; j++) {
                int comp_idx = m * BLOCK_WIDTH + j;
                if (comp_idx < N) {
                    if (a_sh[j] < value || (a_sh[j] == value && comp_idx < i))
                        count++;
                }
            }
            __syncthreads();
        }
        a_sorted[count] = value;
    }
}
```

**Optimizations:**
1. Tiled memory access pattern
2. Shared memory utilization
3. Register caching of `a[i]`
4. Boundary condition handling

---

#### Problem D: π Estimation (Monte Carlo)
**Task:** Estimate π using Monte Carlo method with 1 billion tosses.

**Solution:**
```c
__global__ void pi_estimation(int* block_counts, int nTossesPerThread) {
    __shared__ int partial_counts[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    unsigned long seed = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize random number generator
    curandState_t state;
    curand_init(clock64(), seed, 0, &state);
    
    // Count hits in circle
    int count = 0;
    for (int i = 0; i < nTossesPerThread; i++) {
        float x = 2.0f * curand_uniform(&state) - 1.0f;
        float y = 2.0f * curand_uniform(&state) - 1.0f;
        if (x*x + y*y <= 1.0f) count++;
    }
    partial_counts[tid] = count;
    
    // Block-level reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            partial_counts[tid] += partial_counts[tid + stride];
        }
    }
    
    if (tid == 0) {
        block_counts[blockIdx.x] = partial_counts[0];
    }
}
```

**Implementation Notes:**
- Uses CUDA's `curand` for random numbers
- Two-level reduction (block → global)
- Each thread handles portion of tosses
- Final reduction on host

---

#### Problem E: Find Brightest Pixel
**Task:** Find max value in 1024×1024 grayscale image.

**Solution:**
```c
__global__ void find_max_pixel(float* image, float* block_maxima) {
    __shared__ float partial_max[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load initial value
    partial_max[tid] = (idx < 1024*1024) ? image[idx] : -FLT_MAX;
    
    // Block-level max reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            partial_max[tid] = fmaxf(partial_max[tid], partial_max[tid + stride]);
        }
    }
    
    if (tid == 0) {
        block_maxima[blockIdx.x] = partial_max[0];
    }
}
```

**Key Techniques:**
- 2D image treated as 1D array
- Parallel reduction for finding max
- Shared memory for block-level results
- Final max computed on host

---

#### Problem F: Initialize Array with Dependency
**Task:** Parallelize `a[i] = a[i-1] + i` for 1M elements.

**Solution:**
```c
__global__ void init_array(int* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        a[i] = i * (i + 1) / 2;  // Closed-form solution
    }
}
```

**Optimization Insight:**
- Original loop had dependency: `a[i] = a[i-1] + i`
- Mathematical reformulation: `a[i] = ∑k=1→i k = i(i+1)/2`
- Transforms into embarrassingly parallel problem

---

### Key CUDA Concepts Demonstrated
1. **Memory Hierarchy**
    - Global vs shared vs register usage
    - Coalesced memory access patterns

2. **Parallel Patterns**
    - Map (embarrassingly parallel)
    - Reduction (max, sum)
    - Tiled algorithms

3. **Optimization Techniques**
    - Shared memory for data reuse
    - Register caching
    - Parallel algorithm redesign

4. **Random Number Generation**
    - `curand` library usage
    - Thread-safe RNG initialization

5. **Synchronization**
    - `__syncthreads()` for block coordination
    - Atomic operations (implicit in abort())

6. **Resource Management**
    - Block/grid dimensioning
    - Occupancy considerations
