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
//#include "kernel.h"

#include <sys/stat.h>
#include <string>
#include <fstream>

__device__ u32 fileEncryptionTotalG = 0;

// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void fileEncryption128counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G,  
	u32* encryptionCountG, u32* threadCountG) {

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

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init, s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 pt2Max, pt3Max, threadCount = *threadCountG;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += *encryptionCountG;
	pt2Max = threadRangeStart >> 32;
	pt3Max = threadRangeStart & 0xFFFFFFFF;

	// Initialize plaintext
	pt3Init += threadIndex;
	if (pt3Init < threadIndex) {
		pt2Init++;
	}

	if (pt2Init >= pt2Max && pt3Init >= pt3Max) {
		return;
	}

	// Initialize ciphertext index
	u32 ctIndex = threadIndex*4;

	//if (threadIndex == 0) {
	//	printf("Boundry: %08x %08x\n", pt2Max, pt3Max);
	//	printf("threadCount: %08x\n", threadCount);
	//	printf("encryptionCountG: %08x\n", *encryptionCountG);
	//}

	for (;;) {

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

		// Allocate ciphertext
		ct[ctIndex    ] = s0;
		ct[ctIndex + 1] = s1;
		ct[ctIndex + 2] = s2;
		ct[ctIndex + 3] = s3;

		//if (pt3Init+1 == 0x05ea2a80) {
		//	printf("-------------------------------\n");
		//	printf("threadIndex : %d\n", threadIndex);
		//	printf("Plaintext   : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		//	printf("Ciphertext  : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		//	printf("Ciphertext index  : %d %d %d %d\n", ctIndex, ctIndex + 1, ctIndex + 2, ctIndex + 3);
		//	printf("-------------------------------\n");
		//}

		// Increase plaintext
		pt3Init += threadCount;
		if (pt3Init < threadCount) {
			pt2Init++;
		}

		// Ciphertext index
		ctIndex += threadCount * 4;

		atomicAdd(&fileEncryptionTotalG, 1);

		if (pt2Init >= pt2Max && pt3Init >= pt3Max) {
			break;
		}

	}

	if (threadIndex == 0) {
		//printf("threadIndex : %d\n", threadIndex);
		//printf("Plaintext   : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		//printf("Ciphertext  : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		//printf("Ciphertext index  : %d %d %d %d\n", ctIndex, ctIndex+1, ctIndex+2, ctIndex+3);
		printf("-------------------------------\n");
	}

}


// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void fileEncryption192counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G,
	u32* encryptionCountG, u32* threadCountG) {

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

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 pt2Max, pt3Max, threadCount = *threadCountG;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += *encryptionCountG;
	pt2Max = threadRangeStart >> 32;
	pt3Max = threadRangeStart & 0xFFFFFFFF;

	// Initialize plaintext
	pt3Init += threadIndex;
	if (pt3Init < threadIndex) {
		pt2Init++;
	}

	if (pt2Init >= pt2Max && pt3Init >= pt3Max) {
		return;
	}

	// Initialize ciphertext index
	u32 ctIndex = threadIndex * 4;

	for (;;) {

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

		// Allocate ciphertext
		ct[ctIndex] = s0;
		ct[ctIndex + 1] = s1;
		ct[ctIndex + 2] = s2;
		ct[ctIndex + 3] = s3;

		// Increase plaintext
		pt3Init += threadCount;
		if (pt3Init < threadCount) {
			pt2Init++;
		}

		// Ciphertext index
		ctIndex += threadCount * 4;

		atomicAdd(&fileEncryptionTotalG, 1);

		if (pt2Init >= pt2Max && pt3Init >= pt3Max) {
			break;
		}
	}

	if (threadIndex == 0) {
		//printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		//printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}

}


// CTR encryption with one table extended as 32 columns
// 1 Table [256][32] -> arithmetic shift: __byte_perm function
// SBox[256] is partly expanded
__global__ void fileEncryption256counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G,
	u32* encryptionCountG, u32* threadCountG) {

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

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 pt2Max, pt3Max, threadCount = *threadCountG;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += *encryptionCountG;
	pt2Max = threadRangeStart >> 32;
	pt3Max = threadRangeStart & 0xFFFFFFFF;

	// Initialize plaintext
	pt3Init += threadIndex;
	if (pt3Init < threadIndex) {
		pt2Init++;
	}

	if (pt2Init >= pt2Max && pt3Init >= pt3Max) {
		return;
	}

	// Initialize ciphertext index
	u32 ctIndex = threadIndex * 4;

	for (;;) {

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

		// Allocate ciphertext
		ct[ctIndex] = s0;
		ct[ctIndex + 1] = s1;
		ct[ctIndex + 2] = s2;
		ct[ctIndex + 3] = s3;

		// Increase plaintext
		pt3Init += threadCount;
		if (pt3Init < threadCount) {
			pt2Init++;
		}

		// Ciphertext index
		ctIndex += threadCount * 4;

		atomicAdd(&fileEncryptionTotalG, 1);

		if (pt2Init >= pt2Max && pt3Init >= pt3Max) {
			break;
		}
	}

	if (threadIndex == 0) {
		printf("-------------------------------\n");
	}

}


__host__ int mainFileEncryption() {
	printf("\n");
	printf("########## AES CTR File Encryption Implementation ##########\n");
	printf("\n");

	int chunkSize = 1024;
	const std::string filePath = "C://file-encryption-test//william3.mp4";

	std::fstream fileIn(filePath, std::fstream::in | std::fstream::binary);
	if (fileIn) {

		// Get file size
		fileIn.seekg(0, fileIn.end);
		int fileSize = fileIn.tellg();
		fileIn.seekg(0, fileIn.beg);
		printf("File: %s\n", filePath.c_str());
		printf("Size in bytes: %d\n", fileSize);
		printf("-------------------------------\n");

		// Allocate plaintext and every round key
		u32 *pt, *rk, *roundKeys, *rk192, *roundKeys192, *rk256, *roundKeys256;
		gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
		gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
		gpuErrorCheck(cudaMallocManaged(&roundKeys, TABLE_BASED_KEY_LIST_ROW_SIZE * sizeof(u32)));
		gpuErrorCheck(cudaMallocManaged(&rk192, 6 * sizeof(u32)));
		gpuErrorCheck(cudaMallocManaged(&roundKeys192, TABLE_BASED_KEY_LIST_SIZE_192 * sizeof(u32)));
		gpuErrorCheck(cudaMallocManaged(&rk256, 8 * sizeof(u32)));
		gpuErrorCheck(cudaMallocManaged(&roundKeys256, TABLE_BASED_KEY_LIST_SIZE_256 * sizeof(u32)));

		pt[0] = 0x3243F6A8U;
		pt[1] = 0x885A308DU;
		pt[2] = 0x313198A2U;
		pt[3] = 0x00000000U;

		rk[0] = 0x2B7E1516U;
		rk[1] = 0x28AED2A6U;
		rk[2] = 0xABF71588U;
		rk[3] = 0x09CF4F3CU;

		rk192[0] = 0x8e73b0f7U;
		rk192[1] = 0xda0e6452U;
		rk192[2] = 0xc810f32bU;
		rk192[3] = 0x809079e5U;
		rk192[4] = 0x62f8ead2U;
		rk192[5] = 0x522c6b7bU;

		rk256[0] = 0x603deb10U;
		rk256[1] = 0x15ca71beU;
		rk256[2] = 0x2b73aef0U;
		rk256[3] = 0x857d7781U;
		rk256[4] = 0x1f352c07U;
		rk256[5] = 0x3b6108d7U;
		rk256[6] = 0x2d9810a3U;
		rk256[7] = 0x0914dff4U;

		// Allocate RCON values
		u32* rcon;
		gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
		for (int i = 0; i < RCON_SIZE; i++) {
			rcon[i] = RCON32[i];
		}

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

		// Calculate encryption boundary
		u32 *ct, *encryptionCount, *threadCount;
		gpuErrorCheck(cudaMallocManaged(&threadCount, 1 * sizeof(u32)));
		gpuErrorCheck(cudaMallocManaged(&encryptionCount, 1 * sizeof(u32)));
		threadCount[0] = BLOCKS * THREADS;
		double totalBlockSize = (double)fileSize / BYTE_COUNT;
		encryptionCount[0] = ceil(totalBlockSize);
		u32 ciphertextSize = encryptionCount[0] * U32_SIZE * sizeof(u32);
		gpuErrorCheck(cudaMallocManaged(&ct, ciphertextSize));

		printf("Blocks                        : %d\n", BLOCKS);
		printf("Threads                       : %d\n", THREADS);
		printf("Total thread count            : %d\n", threadCount[0]);
		printf("Total encryptions             : %d\n", encryptionCount[0]);
		printf("Total encryptions in byte     : %d\n", ciphertextSize);
		printf("Each thread encryptions       : %.2f\n", encryptionCount[0] / (double)threadCount[0]);
		printf("-------------------------------\n");
		printf("Initial Counter               : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
		printf("Initial Key                   : %08x %08x %08x %08x\n", rk[0], rk[1], rk[2], rk[3]);
		//printf("Initial Key                    : %08x %08x %08x %08x %08x %08x\n", rk192[0], rk192[1], rk192[2], rk192[3], rk192[4], rk192[5]);
		//printf("Initial Key                    : %08x %08x %08x %08x %08x %08x %08x %08x\n", rk256[0], rk256[1], rk256[2], rk256[3], rk256[4], rk256[5], rk256[6], rk256[7]);
		printf("-------------------------------\n");

		// Key expansion
		keyExpansion(rk, roundKeys);
		keyExpansion192(rk192, roundKeys192);
		keyExpansion256(rk256, roundKeys256);

		clock_t beginTime = clock();
		// Kernels
		fileEncryption128counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, ct, roundKeys, t0, t4, encryptionCount, threadCount);
		//fileEncryption192counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, ct, roundKeys, t0, t4, encryptionCount, threadCount);
		//fileEncryption256counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, ct, roundKeys, t0, t4, encryptionCount, threadCount);
		
		cudaDeviceSynchronize();
		printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
		printf("-------------------------------\n");
		printLastCUDAError();

		u32 totEncryption;
		cudaMemcpyFromSymbol(&totEncryption, fileEncryptionTotalG, sizeof(u32));
		printf("Total encryptions : %I64d\n", totEncryption);
		printf("-------------------------------\n");

		beginTime = clock();
		// Open output file
		const std::string outFilePath = filePath + "_ENC";
		printf("Encrypted File: %s\n", outFilePath.c_str());
		printf("-------------------------------\n");
		std::fstream fileOut(outFilePath, std::fstream::out | std::fstream::binary);
		// Allocate file buffer
		char * buffer = new char[chunkSize];
		while (1) {
			// Read data as a block into buffer:
			fileIn.read(buffer, chunkSize);
			// Decide whether buffer is at the last part
			long readByte = 0;
			if (fileIn) {
				// All characters read successfully
				readByte = chunkSize;
			} else {
				// Only readByte characters could be read
				readByte = fileIn.gcount();
			}
			// Process current buffer
			u32 readInt = 0;
			for (int bufferIndex = 0; bufferIndex < readByte; bufferIndex++) {
				// Process 4 byte as integers
				int bufferIntIndex = (bufferIndex + 1) % U32_SIZE;
				if (bufferIntIndex == 0) {
					// Change 4 byte to int
					readInt = 0;
					readInt |= (0x000000FF & buffer[bufferIndex - 3]) << 24;
					readInt |= (0x000000FF & buffer[bufferIndex - 2]) << 16;
					readInt |= (0x000000FF & buffer[bufferIndex - 1]) << 8;
					readInt |= (0x000000FF & buffer[bufferIndex    ]);
					// XOR with ciphertext
					readInt ^= ct[bufferIndex / U32_SIZE];
					// Change 4 byte back to char
					buffer[bufferIndex - 3] = readInt >> 24;
					buffer[bufferIndex - 2] = readInt >> 16;
					buffer[bufferIndex - 1] = readInt >> 8;
					buffer[bufferIndex] = readInt;
				} else if (bufferIndex == readByte - 1) {
					// Change bufferIntIndex byte to int
					readInt = 0;
					for (int extraByteIndex = 0; extraByteIndex < bufferIntIndex; extraByteIndex++) {
						readInt |= (0x000000FF & buffer[bufferIndex - bufferIntIndex + extraByteIndex + 1]) << ((U32_SIZE -1 -extraByteIndex) * 8);
					}
					// XOR with ciphertext
					readInt ^= ct[bufferIndex / U32_SIZE];
					// Change bufferIntIndex byte back to char
					for (int extraByteIndex = 0; extraByteIndex < bufferIntIndex; extraByteIndex++) {
						buffer[bufferIndex - bufferIntIndex + extraByteIndex + 1] = readInt >> (U32_SIZE - 1 - extraByteIndex) * 8;
					}
				}
			}
			// Write buffer to output file
			fileOut.write(buffer, readByte);
			// stop
			if (readByte < chunkSize) {
				break;
			}
		}
		printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
		printf("-------------------------------\n");

		delete[] buffer;
		fileOut.close();

		// Free alocated arrays
		cudaFree(threadCount);
		cudaFree(encryptionCount);
		cudaFree(ct);
		cudaFree(pt);
		cudaFree(rk);
		cudaFree(roundKeys);
		cudaFree(rk192);
		cudaFree(roundKeys192);
		cudaFree(rk256);
		cudaFree(roundKeys256);
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
	} else {
		printf("File could not be opened: %s\n", filePath.c_str());
	}

	fileIn.close();
	return 0;
}