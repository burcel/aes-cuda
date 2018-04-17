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
__device__ u32 rk3Max = 0;
__device__ u32 totalThreadCount = 0;
__device__ u32 totalEncryptions = 0;
__device__ u32 maxThreadIndex = 0;
#endif // INFO

//__device__ void keyExpansion(u8* key, u32* rk) {
//
//	printf("keyExpansion\n");
//
//	rk[0] = ((u32)key[0] << 24) ^ ((u32)key[1] << 16) ^ ((u32)key[2] << 8) ^ ((u32)key[3]);
//	rk[1] = ((u32)key[4] << 24) ^ ((u32)key[5] << 16) ^ ((u32)key[6] << 8) ^ ((u32)key[7]);
//	rk[2] = ((u32)key[8] << 24) ^ ((u32)key[9] << 16) ^ ((u32)key[10] << 8) ^ ((u32)key[11]);
//	rk[3] = ((u32)key[12] << 24) ^ ((u32)key[13] << 16) ^ ((u32)key[14] << 8) ^ ((u32)key[15]);
//
//	//printf("-- Round 0:\n");
//	//printf("%08x\n", rk[0]);
//	//printf("%08x\n", rk[1]);
//	//printf("%08x\n", rk[2]);
//	//printf("%08x\n", rk[3]);
//
//	for (int rc = 0; rc < ROUND_COUNT; rc++) {
//		u32 temp = rk[rc * 4 + 3];
//		rk[rc * 4 + 4] = rk[rc * 4] ^
//		(T4[(temp >> 16) & 0xff] & 0xff000000) ^
//		(T4[(temp >> 8) & 0xff] & 0x00ff0000) ^
//		(T4[(temp) & 0xff] & 0x0000ff00) ^
//		(T4[(temp >> 24)] & 0x000000ff) ^
//		RCON32[rc];
//		rk[rc * 4 + 5] = rk[rc * 4 + 1] ^ rk[rc * 4 + 4];
//		rk[rc * 4 + 6] = rk[rc * 4 + 2] ^ rk[rc * 4 + 5];
//		rk[rc * 4 + 7] = rk[rc * 4 + 3] ^ rk[rc * 4 + 6];
//
//		//printf("-- Round %d:\n", rc + 1);
//		//printf("%08x\n", rk[rc * 4 + 4]);
//		//printf("%08x\n", rk[rc * 4 + 5]);
//		//printf("%08x\n", rk[rc * 4 + 6]);
//		//printf("%08x\n", rk[rc * 4 + 7]);
//	}
//
//}
//
//__global__ void enc(u8* key, u8* plainTextInput) {
//
//	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//
//	printf("Thread index: %d\n", threadIndex);
//
//	// Create round keys
//	__shared__ u32 rk[TABLE_BASED_KEY_LIST_ROW_SIZE];
//	if (threadIndex == 0) {
//		keyExpansion(key, rk);
//	}
//
//	// Wait until key schedule is finished
//	__syncthreads();
//
//	//printf("\n");
//	//printf("## Starting ##\n");
//	//printf("\n");
//
//	// Create plaintext as 32 bit unsigned integers
//	u32 s0, s1, s2, s3;
//	s0 = ((u32)plainTextInput[0] << 24) |
//		((u32)plainTextInput[1] << 16) |
//		((u32)plainTextInput[2] << 8) |
//		((u32)plainTextInput[3]);
//
//	s1 = ((u32)plainTextInput[4] << 24) |
//		((u32)plainTextInput[5] << 16) |
//		((u32)plainTextInput[6] << 8) |
//		((u32)plainTextInput[7]);
//
//	s2 = ((u32)plainTextInput[8] << 24) |
//		((u32)plainTextInput[9] << 16) |
//		((u32)plainTextInput[10] << 8) |
//		((u32)plainTextInput[11]);
//
//	s3 = ((u32)plainTextInput[12] << 24) |
//		((u32)plainTextInput[13] << 16) |
//		((u32)plainTextInput[14] << 8) |
//		((u32)plainTextInput[15]);
//	
//	// First round just XORs input with key.
//	s0 = s0 ^ rk[0];
//	s1 = s1 ^ rk[1];
//	s2 = s2 ^ rk[2];
//	s3 = s3 ^ rk[3];
//
//	u32 t0, t1, t2, t3;
//	for (int roundCount = 1; roundCount < ROUND_COUNT; roundCount++) {
//		t0 = T0[s0 >> 24] ^ T1[(s1 >> 16) & 0xFF] ^ T2[(s2 >> 8) & 0xFF] ^ T3[s3 & 0xFF] ^ rk[roundCount * 4 + 0];
//		t1 = T0[s1 >> 24] ^ T1[(s2 >> 16) & 0xFF] ^ T2[(s3 >> 8) & 0xFF] ^ T3[s0 & 0xFF] ^ rk[roundCount * 4 + 1];
//		t2 = T0[s2 >> 24] ^ T1[(s3 >> 16) & 0xFF] ^ T2[(s0 >> 8) & 0xFF] ^ T3[s1 & 0xFF] ^ rk[roundCount * 4 + 2];
//		t3 = T0[s3 >> 24] ^ T1[(s0 >> 16) & 0xFF] ^ T2[(s1 >> 8) & 0xFF] ^ T3[s2 & 0xFF] ^ rk[roundCount * 4 + 3];
//
//		s0 = t0;
//		s1 = t1;
//		s2 = t2;
//		s3 = t3;
//
//		//printf("-- Round: %d\n", roundCount);
//		//printf("%08x\n", s0);
//		//printf("%08x\n", s1);
//		//printf("%08x\n", s2);
//		//printf("%08x\n", s3);
//		//printf("-- Round Key\n");
//		//printf("%08x\n", rk[roundCount * 4 + 0]);
//		//printf("%08x\n", rk[roundCount * 4 + 1]);
//		//printf("%08x\n", rk[roundCount * 4 + 2]);
//		//printf("%08x\n", rk[roundCount * 4 + 3]);
//	}
//
//	// Last round uses s-box directly and XORs to produce output.
//	s0 = (T4[t0 >> 24] & 0xFF000000) ^ (T4[(t1 >> 16) & 0xff] & 0x00FF0000) ^ (T4[(t2 >> 8) & 0xff] & 0x0000FF00) ^ (T4[(t3) & 0xFF] & 0x000000FF) ^ rk[40];
//	s1 = (T4[t1 >> 24] & 0xFF000000) ^ (T4[(t2 >> 16) & 0xff] & 0x00FF0000) ^ (T4[(t3 >> 8) & 0xff] & 0x0000FF00) ^ (T4[(t0) & 0xFF] & 0x000000FF) ^ rk[41];
//	s2 = (T4[t2 >> 24] & 0xFF000000) ^ (T4[(t3 >> 16) & 0xff] & 0x00FF0000) ^ (T4[(t0 >> 8) & 0xff] & 0x0000FF00) ^ (T4[(t1) & 0xFF] & 0x000000FF) ^ rk[42];
//	s3 = (T4[t3 >> 24] & 0xFF000000) ^ (T4[(t0 >> 16) & 0xff] & 0x00FF0000) ^ (T4[(t1 >> 8) & 0xff] & 0x0000FF00) ^ (T4[(t2) & 0xFF] & 0x000000FF) ^ rk[43];
//
//	//printf("-- Round: %d\n", 10);
//	//printf("%08x\n", s0);
//	//printf("%08x\n", s1);
//	//printf("%08x\n", s2);
//	//printf("%08x\n", s3);
//	//printf("-- Round Key\n");
//	//printf("%08x\n", rk[40]);
//	//printf("%08x\n", rk[41]);
//	//printf("%08x\n", rk[42]);
//	//printf("%08x\n", rk[43]);
//
//	// Create ciphertext as byte array from 32 bit unsigned integers
//	//u8 cipherText[16];
//	//cipherText[0] = s0 >> 24;
//	//cipherText[1] = (s0 >> 16) & 0xff;
//	//cipherText[2] = (s0 >> 8) & 0xff;
//	//cipherText[3] = s0 & 0xff;
//	//cipherText[4] = s1 >> 24;
//	//cipherText[5] = (s1 >> 16) & 0xff;
//	//cipherText[6] = (s1 >> 8) & 0xff;
//	//cipherText[7] = s1 & 0xff;
//	//cipherText[8] = s2 >> 24;
//	//cipherText[9] = (s2 >> 16) & 0xff;
//	//cipherText[10] = (s2 >> 8) & 0xff;
//	//cipherText[11] = s2 & 0xff;
//	//cipherText[12] = s3 >> 24;
//	//cipherText[13] = (s3 >> 16) & 0xff;
//	//cipherText[14] = (s3 >> 8) & 0xff;
//	//cipherText[15] = s3 & 0xff;
//}

__global__ void exhaustiveSearch(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t1G, u32* t2G, u32* t3G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE];
	__shared__ u32 t1S[TABLE_SIZE];
	__shared__ u32 t2S[TABLE_SIZE];
	__shared__ u32 t3S[TABLE_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ptS[U32_SIZE];
	__shared__ u32 ctS[U32_SIZE];
	__shared__ u32 rkS[U32_SIZE];
	__shared__ u32 threadRange;

	if (threadIdx.x < TABLE_SIZE) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t1S[threadIdx.x] = t1G[threadIdx.x];
		t2S[threadIdx.x] = t2G[threadIdx.x];
		t3S[threadIdx.x] = t3G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];

		if (threadIdx.x == 0) {
			threadRange = *range;
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ptS[threadIdx.x] = pt[threadIdx.x];
			ctS[threadIdx.x] = ct[threadIdx.x];
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

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rkS[0];
		rk1 = rkS[1];
		rk2 = rkS[2];
		rk3 = rkS[3];

		// Create key as 32 bit unsigned integers
		rk3 += threadIndex * threadRange + rangeCount;

		#ifdef  INFO
		if (threadIndex == 0) {
			atomicAdd(&rk3Max, 1);
		}
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = ptS[0];
		s1 = ptS[1];
		s2 = ptS[2];
		s3 = ptS[3];

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
		for (u8 roundCount = 1; roundCount < ROUND_COUNT; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			// TODO: temp & 0xff000000
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >>  8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp      ) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)       ] & 0x000000ff) ^
				rconS[roundCount - 1];
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
			rconS[ROUND_COUNT - 1];
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
						printf("! FOUND KEY\n");
						printf("! Found key : %08x%08x%08x%08x\n", rkS[0], rkS[1], rkS[2], threadIndex * threadRange + rangeCount);
					}
				}
			}
		}
	}
}

__global__ void exhaustiveSearchWithOneTable(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u32* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE];
	__shared__ u32 t4S[TABLE_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ptS[U32_SIZE];
	__shared__ u32 ctS[U32_SIZE];
	__shared__ u32 rkS[U32_SIZE];
	__shared__ u32 threadRange;

	if (threadIdx.x < TABLE_SIZE) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];

		if (threadIdx.x == 0) {
			threadRange = *range;
		}

		if (threadIdx.x < RCON_SIZE) {
			rconS[threadIdx.x] = rconG[threadIdx.x];
		}

		if (threadIdx.x < U32_SIZE) {
			ptS[threadIdx.x] = pt[threadIdx.x];
			ctS[threadIdx.x] = ct[threadIdx.x];
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

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		u32 rk0, rk1, rk2, rk3;
		rk0 = rkS[0];
		rk1 = rkS[1];
		rk2 = rkS[2];
		rk3 = rkS[3];

		// Create key as 32 bit unsigned integers
		rk3 += threadIndex * threadRange + rangeCount;

		#ifdef  INFO
		if (threadIndex == 0) {
			atomicAdd(&rk3Max, 1);
		}
		atomicAdd(&totalEncryptions, 1);
		#endif // INFO

		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = ptS[0];
		s1 = ptS[1];
		s2 = ptS[2];
		s3 = ptS[3];

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
		for (u8 roundCount = 1; roundCount < ROUND_COUNT; roundCount++) {

			// Calculate round key
			u32 temp = rk3;
			// TODO: temp & 0xff000000
			rk0 = rk0 ^
				(t4S[(temp >> 16) & 0xff] & 0xff000000) ^
				(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
				(t4S[(temp) & 0xff] & 0x0000ff00) ^
				(t4S[(temp >> 24)] & 0x000000ff) ^
				rconS[roundCount - 1];
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
			(t4S[(temp >> 8) & 0xff] & 0x00ff0000) ^
			(t4S[(temp) & 0xff] & 0x0000ff00) ^
			(t4S[(temp >> 24)] & 0x000000ff) ^
			rconS[ROUND_COUNT - 1];
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
						printf("! FOUND KEY\n");
						printf("! Found key : %08x%08x%08x%08x\n", rkS[0], rkS[1], rkS[2], threadIndex * threadRange + rangeCount);
					}
				}
			}
		}
	}
}

int main() {

	// Allocate key
	u32* rk;
	gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
	//rk[0] = 0x2B7E1516U;
	//rk[1] = 0x28AED2A6U;
	//rk[2] = 0xABF71588U;
	//rk[3] = 0x09CF4F3CU;
	rk[0] = 0x00000000U;
	rk[1] = 0x00000000U;
	rk[2] = 0x00000000U;
	rk[3] = 0x00000000U;

	// Allocate plaintext
	u32* pt;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	pt[0] = 0x3243F6A8U;
	pt[1] = 0x885A308DU;
	pt[2] = 0x313198A2U;
	pt[3] = 0xE0370734U;

	// Allocate ciphertext
	u32* ct;
	gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
	ct[0] = 0x4390c373U;
	ct[1] = 0xd11979acU;
	ct[2] = 0x6236104cU;
	ct[3] = 0xa3d85b88U;

	// Allocate Tables
	u32 *t0, *t1, *t2, *t3, *t4;
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];
		t1[i] = T1[i];
		t2[i] = T2[i];
		t3[i] = T3[i];
		t4[i] = T4[i];
	}

	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rcon[i] = RCON32[i];
	}

	// Calculate range
	u32* range;
	gpuErrorCheck(cudaMallocManaged(&range, 1 * sizeof(u32)));
	int blocks = 1024;
	int threads = 1024;
	int threadCount = blocks * threads;
	int twoPowerRange = 30;
	double keyRange = pow(2, twoPowerRange);
	double threadRange = keyRange / threadCount;
	range[0] = ceil(threadRange);

	// Printing info
	printf("------------------------------------\n");
	printf("Blocks                             : %d\n", blocks);
	printf("Threads                            : %d\n", threads);
	printf("Total Thread count                 : %d\n", threadCount);
	printf("Key Range (power)                  : %d\n", twoPowerRange);
	printf("Key Range (decimal)                : %.0f\n", keyRange);
	printf("Each Thread Key Range              : %.2f\n", threadRange);
	printf("Each Thread Key Range (kernel)     : %d\n", range[0]);
	printf("Total encryptions                  : %d\n", range[0] * threadCount);
	printf("------------------------------------\n");
	printf("Initial Key                        : %08x%08x%08x%08x\n", rk[0], rk[1], rk[2], rk[3]);
	printf("Plaintext                          : %08x%08x%08x%08x\n", pt[0], pt[1], pt[2], pt[3]);
	printf("Ciphertext                         : %08x%08x%08x%08x\n", ct[0], ct[1], ct[2], ct[3]);
	printf("------------------------------------\n");

	clock_t beginTime = clock();
	//enc<<<1, 1>>>(BLOCKS, THREADS);

	//exhaustiveSearch<<<blocks, threads>>>(pt, ct, rk, t0, t1, t2, t3, t4, rcon, range);

	exhaustiveSearchWithOneTable<<<blocks, threads>>>(pt, ct, rk, t0, t4, rcon, range);

	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);

	// Printing info
	#ifdef  INFO
	printf("------------------------------------\n");
	u32 total;
	cudaMemcpyFromSymbol(&total, totalThreadCount, sizeof(u32));
	printf("Total Thread count                 : %d\n", total);
	cudaMemcpyFromSymbol(&total, maxThreadIndex, sizeof(u32));
	printf("Max thread index                   : %d\n", total);
	cudaMemcpyFromSymbol(&total, rk3Max, sizeof(u32));
	printf("Total Thread Key Range             : %d\n", total);
	cudaMemcpyFromSymbol(&total, totalEncryptions, sizeof(u32));
	printf("Total encryptions                  : %d\n", total);
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
	cudaFree(rcon);
	cudaFree(range);
	
	return 0;
}