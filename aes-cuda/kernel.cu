// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <ctime>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <device_launch_parameters.h>
#include <device_functions.h>

// Custom header 
#include "kernel.h"

//#define INFO 1
#ifdef  INFO
__device__ u32 totalThreadCount = 0;
__device__ u64 totalEncryptions = 0;
__device__ u32 maxThreadIndex = 0;
#endif // INFO

// Basic exhaustive search
// 4 Tables
__global__ void exhaustiveSearch(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t1G, u32* t2G, u32* t3G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE];
	__shared__ u32 t1S[TABLE_SIZE];
	__shared__ u32 t2S[TABLE_SIZE];
	__shared__ u32 t3S[TABLE_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t1S[threadIdx.x] = t1G[threadIdx.x];
		t2S[threadIdx.x] = t2G[threadIdx.x];
		t3S[threadIdx.x] = t3G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}
		 
		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		//if (threadIndex == 0 && rangeCount == 0) {
		//	printf("--Round: %d\n", 0);
		//	printf("%08x%08x%08x%08x\n", s0, s1, s2, s3);
		//	printf("-- Round Key\n");
		//	printf("%08x%08x%08x%08x\n", rk0, rk1, rk2, rk3);
		//}

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			// TODO: temp & 0xff000000
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >>  8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp      ) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)       ] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24] ^ t1S[(s1 >> 16) & 0xFF] ^ t2S[(s2 >> 8) & 0xFF] ^ t3S[s3 & 0xFF] ^ rk0;
			t1 = t0S[s1 >> 24] ^ t1S[(s2 >> 16) & 0xFF] ^ t2S[(s3 >> 8) & 0xFF] ^ t3S[s0 & 0xFF] ^ rk1;
			t2 = t0S[s2 >> 24] ^ t1S[(s3 >> 16) & 0xFF] ^ t2S[(s0 >> 8) & 0xFF] ^ t3S[s1 & 0xFF] ^ rk2;
			t3 = t0S[s3 >> 24] ^ t1S[(s0 >> 16) & 0xFF] ^ t2S[(s1 >> 8) & 0xFF] ^ t3S[s2 & 0xFF] ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

			//if (threadIndex == 0 && rangeCount == 0) {
			//	printf("--Round: %d\n", roundCount);
			//	printf("%08x%08x%08x%08x\n", s0, s1, s2, s3);
			//	printf("-- Round Key\n");
			//	printf("%08x%08x%08x%08x\n", rk0, rk1, rk2, rk3);
			//}
		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >>  8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp      ) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)       ] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table
// 1 Table -> arithmetic shift: 2 shift 1 and
__global__ void exhaustiveSearchWithOneTable(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >>  8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp      ) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)       ] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF], 24) ^ rk0;
			t1 = t0S[s1 >> 24] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF], 24) ^ rk1;
			t2 = t0S[s2 >> 24] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF], 24) ^ rk2;
			t3 = t0S[s3 >> 24] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF], 24) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: 2 shift 1 and
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemory(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4S[threadIdx.x] = t4G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4S[threadIdx.x] = t4G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t3) & 0xFF] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t0) & 0xFF] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t1) & 0xFF] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (t4S[(t2) & 0xFF] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
				(t4S[(temp >>  8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
				(t4S[(temp      ) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
				(t4S[(temp >> 24)       ][warpThreadIndexSBox] & 0x000000ff) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
			(t4S[(temp >>  8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
			(t4S[(temp      ) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
			(t4S[(temp >> 24)       ][warpThreadIndexSBox] & 0x000000ff) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// 4 S-box, each shifted
__global__ void exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4_0G, u32* t4_1G, u32* t4_2G, u32* t4_3G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4_0S[TABLE_SIZE];
	__shared__ u32 t4_1S[TABLE_SIZE];
	__shared__ u32 t4_2S[TABLE_SIZE];
	__shared__ u32 t4_3S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4_0S[threadIdx.x] = t4_0G[threadIdx.x];
		t4_1S[threadIdx.x] = t4_1G[threadIdx.x];
		t4_2S[threadIdx.x] = t4_2G[threadIdx.x];
		t4_3S[threadIdx.x] = t4_3G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U32;
	rk3Init = rk3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^ t4_3S[(temp >> 16) & 0xff] ^ t4_2S[(temp >> 8) & 0xff] ^ t4_1S[(temp) & 0xff] ^ t4_0S[(temp >> 24)] ^ rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^ t4_3S[(temp >> 16) & 0xff] ^ t4_2S[(temp >> 8) & 0xff] ^ t4_1S[(temp) & 0xff] ^ t4_0S[(temp >> 24)] ^ rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = t4_3S[t0 >> 24] ^ t4_2S[(t1 >> 16) & 0xff] ^ t4_1S[(t2 >> 8) & 0xff] ^ t4_0S[(t3) & 0xFF] ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = t4_3S[t1 >> 24] ^ t4_2S[(t2 >> 16) & 0xff] ^ t4_1S[(t3 >> 8) & 0xff] ^ t4_0S[(t0) & 0xFF] ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = t4_3S[t2 >> 24] ^ t4_2S[(t3 >> 16) & 0xff] ^ t4_1S[(t0 >> 8) & 0xff] ^ t4_0S[(t1) & 0xFF] ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = t4_3S[t3 >> 24] ^ t4_2S[(t0 >> 16) & 0xff] ^ t4_1S[(t1 >> 8) & 0xff] ^ t4_0S[(t2) & 0xFF] ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk3Init == MAX_U32) {
			rk2Init++;
		}

		// Create key as 32 bit unsigned integers
		rk3Init++;
	}
}

// Key expansion from given key set, populate rk[44]
void keyExpansion(u32* key, u32* rk) {

	u32 rk0, rk1, rk2, rk3;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++) {
		u32 temp = rk3;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk2 ^ rk3;

		rk[roundCount * 4 + 4] = rk0;
		rk[roundCount * 4 + 5] = rk1;
		rk[roundCount * 4 + 6] = rk2;
		rk[roundCount * 4 + 7] = rk3;
	}
}

// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* rk, u32* t0G, u32* t4G, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[TABLE_BASED_KEY_LIST_ROW_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < TABLE_BASED_KEY_LIST_ROW_SIZE) {
			rkS[threadIdx.x] = rk[threadIdx.x];
		}

	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	pt2Init = pt2Init + threadRangeStart / MAX_U32;
	pt3Init = pt3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];
		s1 = s1 ^ rkS[1];
		s2 = s2 ^ rkS[2];
		s3 = s3 ^ rkS[3];

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[40];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[41];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[42];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[43];

		//if (threadIndex == 1048576) {
		//	printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		//}

		// Overflow
		if (pt3Init == MAX_U32) {
			pt2Init++;
		}

		// Create key as 32 bit unsigned integers
		pt3Init++;
	}

	if (threadIndex == 1048575) {
		printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
	}

}

// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// 4 S-box, each shifted
__global__ void counterWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox(u32* pt, u32* rk, u32* t0G, u32* t4_0G, u32* t4_1G, u32* t4_2G, u32* t4_3G, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4_0S[TABLE_SIZE];
	__shared__ u32 t4_1S[TABLE_SIZE];
	__shared__ u32 t4_2S[TABLE_SIZE];
	__shared__ u32 t4_3S[TABLE_SIZE];
	__shared__ u32 rkS[TABLE_BASED_KEY_LIST_ROW_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4_0S[threadIdx.x] = t4_0G[threadIdx.x];
		t4_1S[threadIdx.x] = t4_1G[threadIdx.x];
		t4_2S[threadIdx.x] = t4_2G[threadIdx.x];
		t4_3S[threadIdx.x] = t4_3G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		if (threadIdx.x < TABLE_BASED_KEY_LIST_ROW_SIZE) {
			rkS[threadIdx.x] = rk[threadIdx.x];
		}

	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	pt2Init = pt2Init + threadRangeStart / MAX_U32;
	pt3Init = pt3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];
		s1 = s1 ^ rkS[1];
		s2 = s2 ^ rkS[2];
		s3 = s3 ^ rkS[3];

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = t4_3S[t0 >> 24] ^ t4_2S[(t1 >> 16) & 0xff] ^ t4_1S[(t2 >> 8) & 0xff] ^ t4_0S[(t3) & 0xFF] ^ rkS[40];
		s1 = t4_3S[t1 >> 24] ^ t4_2S[(t2 >> 16) & 0xff] ^ t4_1S[(t3 >> 8) & 0xff] ^ t4_0S[(t0) & 0xFF] ^ rkS[41];
		s2 = t4_3S[t2 >> 24] ^ t4_2S[(t3 >> 16) & 0xff] ^ t4_1S[(t0 >> 8) & 0xff] ^ t4_0S[(t1) & 0xFF] ^ rkS[42];
		s3 = t4_3S[t3 >> 24] ^ t4_2S[(t0 >> 16) & 0xff] ^ t4_1S[(t1 >> 8) & 0xff] ^ t4_0S[(t2) & 0xFF] ^ rkS[43];

		//if (s0 == 0x00000000U) {
		//	printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s1, s3);
		//}

		// Overflow
		if (pt3Init == MAX_U32) {
			pt2Init++;
		}

		// Create key as 32 bit unsigned integers
		pt3Init++;
	}

	if (threadIndex == 1048575) {
		printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
	}
}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void exhaustiveSearch192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];


	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];
	rk4Init = rk[4];
	rk5Init = rk[5];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk4Init = rk4Init + threadRangeStart / MAX_U32;
	rk5Init = rk5Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Calculate round keys
		u32 rk0, rk1, rk2, rk3, rk4, rk5;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;
		rk4 = rk4Init;
		rk5 = rk5Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		u8 rconIndex = 0;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_192; roundCount++) {
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);

			// Add round key
			if (roundCount % 3 == 0) {
				t0 = t0 ^ rk4;
				t1 = t1 ^ rk5;
				// Calculate round key
				u32 temp = rk5;
				rk0 = rk0 ^
					(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
					(t4S[(temp >>  8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
					(t4S[(temp      ) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
					(t4S[(temp >> 24)       ][warpThreadIndexSBox] & 0x000000ff) ^
					rconS[rconIndex++];
				rk1 = rk1 ^ rk0;
				rk2 = rk2 ^ rk1;
				rk3 = rk3 ^ rk2;
				rk4 = rk4 ^ rk3;
				rk5 = rk5 ^ rk4;

				t2 = t2 ^ rk0;
				t3 = t3 ^ rk1;
			} else if (roundCount % 3 == 1) {
				t0 = t0 ^ rk2;
				t1 = t1 ^ rk3;
				t2 = t2 ^ rk4;
				t3 = t3 ^ rk5;
			} else {
				// Calculate round key
				u32 temp = rk5;
				rk0 = rk0 ^
					(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
					(t4S[(temp >>  8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
					(t4S[(temp      ) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
					(t4S[(temp >> 24)       ][warpThreadIndexSBox] & 0x000000ff) ^
					rconS[rconIndex++];
				rk1 = rk1 ^ rk0;
				rk2 = rk2 ^ rk1;
				rk3 = rk3 ^ rk2;
				rk4 = rk4 ^ rk3;
				rk5 = rk5 ^ rk4;

				t0 = t0 ^ rk0;
				t1 = t1 ^ rk1;
				t2 = t2 ^ rk2;
				t3 = t3 ^ rk3;
			}

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;
		}

		// Calculate the last round key
		u32 temp = rk5;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
			(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
			(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
			rconS[rconIndex];

		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk5Init == MAX_U32) {
			rk4Init++;
		}

		// Create key as 32 bit unsigned integers
		rk5Init++;
	}
}

// Key expansion from given key set, populate rk[52]
void keyExpansion192(u32* key, u32* rk) {

	u32 rk0, rk1, rk2, rk3, rk4, rk5;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];
	rk4 = key[4];
	rk5 = key[5];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;
	rk[4] = rk4;
	rk[5] = rk5;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT_192; roundCount++) {
		u32 temp = rk5;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk3 ^ rk2;
		rk4 = rk4 ^ rk3;
		rk5 = rk5 ^ rk4;

		rk[roundCount * 6 + 6] = rk0;
		rk[roundCount * 6 + 7] = rk1;
		rk[roundCount * 6 + 8] = rk2;
		rk[roundCount * 6 + 9] = rk3;
		if (roundCount == 7) {
			break;
		}
		rk[roundCount * 6 + 10] = rk4;
		rk[roundCount * 6 + 11] = rk5;
	}

	// Print keys
	//for (int i = 0;i < 52;i++) {
	//	printf("%08x ", rk[i]);
	//	if ((i+1) % 4 == 0) {
	//		printf("Round: %d\n", i / 4);
	//	}
	//}
}

// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void counter192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* rk, u32* t0G, u32* t4G, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[TABLE_BASED_KEY_LIST_SIZE_192];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < TABLE_BASED_KEY_LIST_SIZE_192) {
			rkS[threadIdx.x] = rk[threadIdx.x];
		}

	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	pt2Init = pt2Init + threadRangeStart / MAX_U32;
	pt3Init = pt3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];
		s1 = s1 ^ rkS[1];
		s2 = s2 ^ rkS[2];
		s3 = s3 ^ rkS[3];

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_192; roundCount++) {

			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[48];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[49];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[50];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[51];

		/*if (threadIndex == 0 && rangeCount == 0) {
			printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		}*/

		// Overflow
		if (pt3Init == MAX_U32) {
			pt2Init++;
		}

		// Create key as 32 bit unsigned integers
		pt3Init++;
	}

	if (threadIndex == 1048575) {
		printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
	}

}

// Exhaustive search with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void exhaustiveSearch256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];


	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];
		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init, rk4Init, rk5Init, rk6Init, rk7Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];
	rk4Init = rk[4];
	rk5Init = rk[5];
	rk6Init = rk[6];
	rk7Init = rk[7];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk6Init = rk6Init + threadRangeStart / MAX_U32;
	rk7Init = rk7Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Calculate round keys
		u32 rk0, rk1, rk2, rk3, rk4, rk5, rk6, rk7;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;
		rk4 = rk4Init;
		rk5 = rk5Init;
		rk6 = rk6Init;
		rk7 = rk7Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		u8 rconIndex = 0;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_256; roundCount++) {
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT);

			// Add round key
			if (roundCount % 2 == 0) {
				t0 = t0 ^ rk4;
				t1 = t1 ^ rk5;
				t2 = t2 ^ rk6;
				t3 = t3 ^ rk7;
			} else {
				// Calculate round key
				u32 temp = rk7;
				rk0 = rk0 ^
					(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
					(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
					(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
					(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
					rconS[rconIndex++];
				rk1 = rk1 ^ rk0;
				rk2 = rk2 ^ rk1;
				rk3 = rk3 ^ rk2;
				rk4 = rk4 ^
					(t4S[(rk3 >> 24) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
					(t4S[(rk3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
					(t4S[(rk3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
					(t4S[(rk3) & 0xff][warpThreadIndexSBox] & 0x000000ff);
				rk5 = rk5 ^ rk4;
				rk6 = rk6 ^ rk5;
				rk7 = rk7 ^ rk6;

				t0 = t0 ^ rk0;
				t1 = t1 ^ rk1;
				t2 = t2 ^ rk2;
				t3 = t3 ^ rk3;
			}

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;
		}

		// Calculate the last round key
		u32 temp = rk7;
		rk0 = rk0 ^
			(t4S[(temp >> 16) & 0xff][warpThreadIndexSBox] & 0xff000000) ^
			(t4S[(temp >> 8) & 0xff][warpThreadIndexSBox] & 0x00ff0000) ^
			(t4S[(temp) & 0xff][warpThreadIndexSBox] & 0x0000ff00) ^
			(t4S[(temp >> 24)][warpThreadIndexSBox] & 0x000000ff) ^
			rconS[rconIndex++];

		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		// Overflow
		if (rk7Init == MAX_U32) {
			rk6Init++;
		}

		// Create key as 32 bit unsigned integers
		rk7Init++;
	}
}

// Key expansion from given key set, populate rk[52]
void keyExpansion256(u32* key, u32* rk) {

	u32 rk0, rk1, rk2, rk3, rk4, rk5, rk6, rk7;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];
	rk4 = key[4];
	rk5 = key[5];
	rk6 = key[6];
	rk7 = key[7];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;
	rk[4] = rk4;
	rk[5] = rk5;
	rk[6] = rk6;
	rk[7] = rk7;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT_256; roundCount++) {
		u32 temp = rk7;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk3 ^ rk2;
		rk4 = rk4 ^ T4_3[(rk3 >> 24) & 0xff] ^ T4_2[(rk3 >> 16) & 0xff] ^ T4_1[(rk3 >> 8) & 0xff] ^ T4_0[rk3 & 0xff];
		rk5 = rk5 ^ rk4;
		rk6 = rk6 ^ rk5;
		rk7 = rk7 ^ rk6;

		rk[roundCount * 8 + 8] = rk0;
		rk[roundCount * 8 + 9] = rk1;
		rk[roundCount * 8 + 10] = rk2;
		rk[roundCount * 8 + 11] = rk3;
		if (roundCount == 6) {
			break;
		}
		rk[roundCount * 8 + 12] = rk4;
		rk[roundCount * 8 + 13] = rk5;
		rk[roundCount * 8 + 14] = rk6;
		rk[roundCount * 8 + 15] = rk7;
		
	}

	//for (int i = 0; i < 60; i++) {
	//	printf("%08x ", rk[i]);
	//	if ((i + 1) % 4 == 0) {
	//		printf("Round: %d\n", i / 4);
	//	}
	//}
}

// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* rk, u32* t0G, u32* t4G, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[TABLE_BASED_KEY_LIST_SIZE_256];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < TABLE_BASED_KEY_LIST_SIZE_256) {
			rkS[threadIdx.x] = rk[threadIdx.x];
		}

	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	pt2Init = pt2Init + threadRangeStart / MAX_U32;
	pt3Init = pt3Init + threadRangeStart % MAX_U32;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];
		s1 = s1 ^ rkS[1];
		s2 = s2 ^ rkS[2];
		s3 = s3 ^ rkS[3];

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_256; roundCount++) {

			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[56];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[57];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[58];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[59];

		//if (threadIndex == 0 && rangeCount == 0) {
		//printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		//}

		// Overflow
		if (pt3Init == MAX_U32) {
			pt2Init++;
		}

		// Create key as 32 bit unsigned integers
		pt3Init++;
	}

	if (threadIndex == 1048575) {
		printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
	}

}


__global__ void smallAesExhaustiveSearch(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[16];
	__shared__ u32 t4S[16];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];

	if (threadIdx.x < 16) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ctS[threadIdx.x] = ct[threadIdx.x];

		}
	}
	// </SHARED MEMORY>

	#ifdef  INFO
	atomicAdd(&totalThreadCount, 1);
	atomicMax(&maxThreadIndex, threadIndex);
	#endif // INFO

	// Wait until every thread is ready
	__syncthreads();

	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];
	rk1Init = rk[1];
	rk2Init = rk[2];
	rk3Init = rk[3];

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = (u64)threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / MAX_U16;
	rk3Init = rk3Init + threadRangeStart % MAX_U16;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		#ifdef  INFO
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;
		rk1 = rk1Init;
		rk2 = rk2Init;
		rk3 = rk3Init;

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rk0;
		s1 = s1 ^ rk1;
		s2 = s2 ^ rk2;
		s3 = s3 ^ rk3;

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				(t4S[(temp >>  8) & 0xf] & 0xf000) ^
				(t4S[(temp >>  4) & 0xf] & 0x0f00) ^
				(t4S[(temp      ) & 0xf] & 0x00f0) ^
				(t4S[(temp >> 12)      ] & 0x000f) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			// Table based round function
			t0 = t0S[s0 >> 12] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xF], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 4) & 0xF], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xF], SHIFT_3_RIGHT) ^ rk0;
			t1 = t0S[s1 >> 12] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xF], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 4) & 0xF], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xF], SHIFT_3_RIGHT) ^ rk1;
			t2 = t0S[s2 >> 12] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xF], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 4) & 0xF], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xF], SHIFT_3_RIGHT) ^ rk2;
			t3 = t0S[s3 >> 12] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xF], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 4) & 0xF], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xF], SHIFT_3_RIGHT) ^ rk3;

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			(t4S[(temp >>  8) & 0xf] & 0xf000) ^
			(t4S[(temp >>  4) & 0xf] & 0x0f00) ^
			(t4S[(temp      ) & 0xf] & 0x00f0) ^
			(t4S[(temp >> 12)      ] & 0x000f) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 12] & 0xF000) ^ (t4S[(t1 >> 8) & 0xf] & 0x0F00) ^ (t4S[(t2 >> 4) & 0xf] & 0x00F0) ^ (t4S[(t3) & 0xF] & 0x000F) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = (t4S[t1 >> 12] & 0xF000) ^ (t4S[(t2 >> 8) & 0xf] & 0x0F00) ^ (t4S[(t3 >> 4) & 0xf] & 0x00F0) ^ (t4S[(t0) & 0xF] & 0x000F) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = (t4S[t2 >> 12] & 0xF000) ^ (t4S[(t3 >> 8) & 0xf] & 0x0F00) ^ (t4S[(t0 >> 4) & 0xf] & 0x00F0) ^ (t4S[(t1) & 0xF] & 0x000F) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = (t4S[t3 >> 12] & 0xF000) ^ (t4S[(t0 >> 8) & 0xf] & 0x0F00) ^ (t4S[(t1 >> 4) & 0xf] & 0x00F0) ^ (t4S[(t2) & 0xF] & 0x000F) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key %d : \n", threadIndex, rk0Init, rk1Init, rk2Init, rk3Init);
					}
				}
			}
		}

		if (threadIndex == 0) {
			printf("! CT %d : %08x %08x %08x %08x\n", threadIndex, s0, s1, s2, s3);
		}

		// Overflow
		if (rk3Init == MAX_U16) {
			rk2Init++;
			rk3Init = 0xFFFFFFFF;
		}

		rk3Init++;
	}
}

int main() {

	// <AES-128>
	// Allocate key
	u32* rk;
	gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
	rk[0] = 0x00000000U;
	rk[1] = 0x00000000U;
	rk[2] = 0x00000000U;
	rk[3] = 0x00000000U;

	// aes-cipher-internals.xlsx
	//rk[0] = 0x2B7E1516U;
	//rk[1] = 0x28AED2A6U;
	//rk[2] = 0xABF71588U;
	//rk[3] = 0x09CF4F3CU;

	// Allocate plaintext
	u32* pt;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	//pt[0] = 0x00000000U;
	//pt[1] = 0x00000000U;
	//pt[2] = 0x00000000U;
	//pt[3] = 0x00000000U;

	// aes-cipher-internals.xlsx
	pt[0] = 0x6bc1bee2U;
	pt[1] = 0x2e409f96U;
	pt[2] = 0xe93d7e11U;
	pt[3] = 0x7393172aU;

	// Allocate ciphertext
	u32* ct;
	gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
	ct[0] = 0xF3EED1BDU;
	ct[1] = 0xB5D2A03CU;
	ct[2] = 0x064B5A7EU;
	ct[3] = 0x3DB181F8U;

	// aes-cipher-internals.xlsx
	//ct[0] = 0x3925841DU;
	//ct[1] = 0x02DC09FBU;
	//ct[2] = 0xDC118597U;
	//ct[3] = 0x196A0B32U;

	// CTR round keys
	u32* roundKeys;
	gpuErrorCheck(cudaMallocManaged(&roundKeys, TABLE_BASED_KEY_LIST_ROW_SIZE * sizeof(u32)));
	// </AES-128>

	// <AES-192>
	u32* rk192;
	gpuErrorCheck(cudaMallocManaged(&rk192, 6 * sizeof(u32)));
	rk192[0] = 0x8e73b0f7U;
	rk192[1] = 0xda0e6452U;
	rk192[2] = 0xc810f32bU;
	rk192[3] = 0x809079e5U;
	rk192[4] = 0x62f8ead2U;
	rk192[5] = 0x522c6b7bU;
	// CTR round keys
	u32* roundKeys192;
	gpuErrorCheck(cudaMallocManaged(&roundKeys192, TABLE_BASED_KEY_LIST_SIZE_192 * sizeof(u32)));
	// </AES-192>

	// <AES-256>
	u32* rk256;
	gpuErrorCheck(cudaMallocManaged(&rk256, 8 * sizeof(u32)));
	rk256[0] = 0x603deb10U;
	rk256[1] = 0x15ca71beU;
	rk256[2] = 0x2b73aef0U;
	rk256[3] = 0x857d7781U;
	rk256[4] = 0x1f352c07U;
	rk256[5] = 0x3b6108d7U;
	rk256[6] = 0x2d9810a3U;
	rk256[7] = 0x0914dff4U;
	// CTR round keys
	u32* roundKeys256;
	gpuErrorCheck(cudaMallocManaged(&roundKeys256, TABLE_BASED_KEY_LIST_SIZE_256 * sizeof(u32)));
	// </AES-192>

	// Allocate Tables
	u32 *t0, *t1, *t2, *t3, *t4, *t4_0, *t4_1, *t4_2, *t4_3;
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];
		t1[i] = T1[i];
		t2[i] = T2[i];
		t3[i] = T3[i];
		t4[i] = T4[i];
		t4_0[i] = T4_0[i];
		t4_1[i] = T4_1[i];
		t4_2[i] = T4_2[i];
		t4_3[i] = T4_3[i];
	}

	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rcon[i] = RCON32[i];
	}

	// -- Small AES --
	rk[0] = 0x00000000U;
	rk[1] = 0x00000000U;
	rk[2] = 0x00000000U;
	rk[3] = 0x0000FFFFU;

	pt[0] = 0x00006cbeU;
	pt[1] = 0x00002e40U;
	pt[2] = 0x0000e93dU;
	pt[3] = 0x00007393U;

	ct[0] = 0x00002ec7U;
	ct[1] = 0x000065c7U;
	ct[2] = 0x00005adcU;
	ct[3] = 0x000094e6U;

	u32 *t0Sml, *t1Sml, *t2Sml, *t3Sml, *t4Sml;
	gpuErrorCheck(cudaMallocManaged(&t0Sml, 16 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1Sml, 16 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2Sml, 16 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3Sml, 16 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4Sml, 16 * sizeof(u32)));
	for (int i = 0; i < 16; i++) {
		t0Sml[i] = T0_SML[i];
		t1Sml[i] = T1_SML[i];
		t2Sml[i] = T2_SML[i];
		t3Sml[i] = T3_SML[i];
		t4Sml[i] = T4_SML[i];
	}

	u32* rconSml;
	gpuErrorCheck(cudaMallocManaged(&rconSml, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rconSml[i] = RCON_SML[i];
	}

	// Calculate range
	u32* range;
	gpuErrorCheck(cudaMallocManaged(&range, 1 * sizeof(u32)));

	int threadCount = BLOCKS * THREADS;
	
	double keyRange = pow(2, TWO_POWER_RANGE);
	double threadRange = keyRange / threadCount;
	*range = ceil(threadRange);

	// Printing info
	printf("------------------------------------\n");
	printf("Blocks                             : %d\n", BLOCKS);
	printf("Threads                            : %d\n", THREADS);
	printf("Total Thread count                 : %d\n", threadCount);
	printf("Key Range (power)                  : %d\n", TWO_POWER_RANGE);
	printf("Key Range (decimal)                : %.0f\n", keyRange);
	printf("Each Thread Key Range              : %.2f\n", threadRange);
	printf("Each Thread Key Range (kernel)     : %d\n", range[0]);
	printf("Total encryptions                  : %.0f\n", ceil(threadRange) * threadCount);
	printf("------------------------------------\n");
	printf("Initial Key                        : %08x %08x %08x %08x\n", rk[0], rk[1], rk[2], rk[3]);
	printf("Plaintext                          : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
	printf("Ciphertext                         : %08x %08x %08x %08x\n", ct[0], ct[1], ct[2], ct[3]);
	printf("------------------------------------\n");

	clock_t beginTime = clock();

	// -- AES-128 Exhaustive Search --

	//exhaustiveSearch<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t1, t2, t3, t4, rcon, range);

	//exhaustiveSearchWithOneTable<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);

	//exhaustiveSearchWithOneTableExtendedSharedMemory<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);

	//exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4, rcon, range);

	//exhaustiveSearchWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS >>>(pt, ct, rk, t0, t4, rcon, range);

	//exhaustiveSearchWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox<<<BLOCKS, THREADS>>>(pt, ct, rk, t0, t4_0, t4_1, t4_2, t4_3, rcon, range);

	// -- AES-128 CTR --

	//keyExpansion(rk, roundKeys);

	//counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, roundKeys, t0, t4, range);

	//counterWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox<<<BLOCKS, THREADS>>>(pt, roundKeys, t0, t4_0, t4_1, t4_2, t4_3, range);

	// -- AES-192 Exhaustive Search --
	//exhaustiveSearch192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, ct, rk192, t0, t4, rcon, range);

	// -- AES-192 CTR --
	//keyExpansion192(rk192, roundKeys192);

	//counter192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, roundKeys192, t0, t4, range);

	// -- AES-256 Exhaustive Search --
	//exhaustiveSearch256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, ct, rk256, t0, t4, rcon, range);

	// -- AES-256 CTR --
	//keyExpansion256(rk256, roundKeys256);

	//counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, roundKeys256, t0, t4, range);

	// -- Small AES --

	smallAesExhaustiveSearch<<<BLOCKS, THREADS>>>(pt, ct, rk, t0Sml, t4Sml, rconSml, range);

	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);

	printLastCUDAError();

	// Printing info
	#ifdef  INFO
	printf("------------------------------------\n");
	u32 total;
	cudaMemcpyFromSymbol(&total, totalThreadCount, sizeof(u32));
	printf("Total Thread count                 : %d\n", total);
	cudaMemcpyFromSymbol(&total, maxThreadIndex, sizeof(u32));
	printf("Max Thread Index                   : %d\n", total);
	u64 totEncryption;
	cudaMemcpyFromSymbol(&totEncryption, totalEncryptions, sizeof(u64));
	printf("Total encryptions                  : %lu\n", totEncryption);
	printf("------------------------------------\n");
	#endif // INFO

	// Free alocated arrays
	cudaFree(rk);
	cudaFree(pt);
	cudaFree(ct);
	cudaFree(t0);
	cudaFree(t1);
	cudaFree(t2);
	cudaFree(t3);
	cudaFree(t4);
	cudaFree(t4_0);
	cudaFree(t4_1);
	cudaFree(t4_2);
	cudaFree(t4_3);
	cudaFree(rcon);
	cudaFree(range);
	cudaFree(t0Sml);
	cudaFree(t1Sml);
	cudaFree(t2Sml);
	cudaFree(t3Sml);
	cudaFree(t4Sml);
	cudaFree(rconSml);
	
	return 0;
}
