### C Programming Fundamentals Exam

#### Multiple Choice Questions

**Q1.** Is this code valid?
```c
char c;
int i = 67;
c = i;
printf("The ASCII character code for \"%c\" is %d\n", c, c);
```
○ True  
○ False

**Q2.** What is the output of this code?
```c
double y = 1.4;
int x = y;
printf("x: %d, y: %f\n", x, y);
```
A. `x: 1, y: 1.4`  
B. `x: 1, y: 1.0`  
C. Compile error

**Q3.** What is the output?
```c
int i = 1, j = 1;
while (i == 1) {
    int j = 2;
    printf("Inside: i=%d, j=%d ", i, j);
    i--;
}
printf("Outside: i=%d, j=%d\n", i, j);
```
A. Inside: i=1,j=2 Outside: i=0,j=1  
B. Inside: i=1,j=1 Outside: i=0,j=1  
C. Inside: i=1,j=2 Outside: i=1,j=1

**Q4.** What does this code print?
```c
int a = 1;
if (a = -2) {
    printf("hello world!\n");
}
```
A. Nothing  
B. `hello world!`  
C. Runtime error

**Q5.** What does `(2 || 4)` evaluate to in C?  
A. 2  
B. 1  
C. 6

**Q6.** What does `(2 | 4)` evaluate to in C?  
A. 2  
B. 1  
C. 6

**Q7.** What is `c` after this code?
```c
unsigned char c = 255;
c += 2;
```
A. 0  
B. 1  
C. 257

**Q8.** What is the value of `k` after execution?
```c
int i = 1, j = 0;
int k = 1 / j;
```
A. 0  
B. Runtime error  
C. Infinity

**Q9.** What is the output?
```c
int i, sum = 0, a[5];
for (i = 0; i < 5; i++) {
    a[i] = 1;
    sum += a[i];
    printf("%d\n", sum);
}
```
A. 1 2 3 4 5  
B. 5 5 5 5 5  
C. Compile error

**Q10.** What is the output?
```c
void foo(int x) { x = 2; printf("During: %d\n", x); }
int main() {
    int i = 1;
    printf("Before: %d\n", i);
    foo(i);
    printf("After: %d\n", i);
}
```
A. Before:1 During:2 After:1  
B. Before:1 During:2 After:2  
C. Before:1 During:1 After:1

**Q11.** What is the output?
```c
int z = 7;
int* zp = &z;
printf("Addr: %p\nVal: %d\nPtrVal: %d\n", (void*)zp, z, *zp);
```
A. [z's address], 7, 7  
B. 7, 7, [z's address]  
C. Runtime error

**Q12.** What is the output?
```c
int x = 10;
int* px = &x;
px++;
printf("%d", *px);
```
A. 10  
B. 11  
C. Undefined

**Q13.** What is the output? (Assume `arr` at address 1000)
```c
int arr[2] = {3, 7};
int* p = arr;
for (int i = 0; i < 2; i++) 
    printf("%d\n", *p++);
```
A. 3, 7  
B. 1000, 1004  
C. 3, 4

**Q14.** What is the type of `P1` in `#define P1 3.141`?  
A. `double`  
B. No type (text substitution)  
C. `float`

**Q15.** Which is the logical OR operator?  
A. `|`  
B. `||`  
C. `OR`

**Q16.** Which function allocates+initializes memory?  
A. `malloc`  
B. `calloc`  
C. `alloc`

**Q17.** Invalid file inclusion:  
A. `#include <file>`  
B. `#include "file"`  
C. `#include file`

**Q18.** Local variables are stored in:  
A. Stack  
B. Heap  
C. Global memory

**Q19.** Standard I/O header:  
A. `<stdio.h>`  
B. `<stdlib.h>`  
C. `<io.h>`

**Q20.** Correct `scanf` for `float a` then `double b`:  
A. `scanf("%f %lf", &a, &b)`  
B. `scanf("%f %f", &a, &b)`  
C. `scanf("%lf %lf", &a, &b)`

---

### Solutions

**Q1:** True
- *Explanation:* Implicit `int`-to-`char` conversion is valid in C.

**Q2:** A
- *Explanation:* `double` to `int` truncates decimals (1.4 → 1).

**Q3:** A
- *Explanation:* Inner `j` shadows outer `j` in loop scope.

**Q4:** B
- *Explanation:* Assignment in `if` evaluates to `-2` (truthy).

**Q5:** B
- *Explanation:* Logical OR returns `1` (true) for non-zero operands.

**Q6:** C
- *Explanation:* Bitwise OR: `2|4 = 6` (010 | 100 = 110).

**Q7:** B
- *Explanation:* Unsigned char wraps (255+2=257 → 257%256=1).

**Q8:** B
- *Explanation:* Division by zero causes runtime error.

**Q9:** A
- *Explanation:* Prints cumulative sum (1, 1+1=2, ..., 5).

**Q10:** A
- *Explanation:* `foo()` gets a copy of `i` (pass-by-value).

**Q11:** A
- *Explanation:* Prints address of `z`, then `z`'s value twice.

**Q12:** C
- *Explanation:* `px++` moves to invalid memory (undefined behavior).

**Q13:** A
- *Explanation:* Pointer traverses array elements `{3,7}`.

**Q14:** B
- *Explanation:* `#define` performs text substitution (no type).

**Q15:** B
- *Explanation:* `||` is logical OR; `|` is bitwise OR.

**Q16:** B
- *Explanation:* `calloc` allocates + zero-initializes memory.

**Q17:** C
- *Explanation:* Must use angle brackets or quotes for `#include`.

**Q18:** A
- *Explanation:* Local variables are stored on the stack.

**Q19:** A
- *Explanation:* `<stdio.h>` contains I/O functions like `printf`.

**Q20:** A
- *Explanation:* `%f` for `float`, `%lf` for `double` in `scanf`.
