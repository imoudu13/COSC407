#include <stdio.h>
#include <stdlib.h>

void printNums(double *a, int n) {
	int x = 5;
	for (int i = 0; i < x; i++)
		printf("a[%d]: %.7f \n", i, a[i]);
	puts("... ");
	for (int i = x; i >= 1; i--)
		printf("a[%d]: %.7f \n", n - i, a[(int)n - i]);
}

int main() {
  const int n = 10000000;
  double* a = (double*) malloc (n * sizeof(double));

  for (int i = 0; i < n; i++) {
    a[i] = (double) i / n;
  }

  printNums(a, n);

  return 0;
}
