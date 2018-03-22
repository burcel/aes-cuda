// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <device_launch_parameters.h>
#include <device_functions.h>

#define SIZE	1024

__global__ void vectorAdd(int *a, int *b, int *c, int n) {

	int i = threadIdx.x;

	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	int *a, *b, *c;

	cudaMallocManaged(&a, SIZE * sizeof(int));
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; i++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	vectorAdd <<< 1, SIZE >>>(a, b, c, SIZE);

	cudaDeviceSynchronize();

	for (int i = 0; i < 10; i++) {
		printf("c[%d] = %d\n", i, c[i]);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}