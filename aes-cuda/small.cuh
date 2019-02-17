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

#define PROB_1
//#define PROB_2
//#define PROB_3
//#define PROB_4

__device__ u64 ciphertextResultG = 0;
__device__ u64 totalEncryptionsG = 0;

__host__ void keyExpansion(u64 key, u64 *roundKeys64) {
	u64 rk0, rk1, rk2, rk3, temp;
	roundKeys64[0] = key;
	printf("key[%02d]                       : %016llx\n", 0, key);
	rk0 = (key >> 48) & 0xFFFF;
	rk1 = (key >> 32) & 0xFFFF;
	rk2 = (key >> 16) & 0xFFFF;
	rk3 = (key >>  0) & 0xFFFF;
	for (u8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++) {
		temp = rk3;
		rk0 = rk0 ^
			(T4_SML[(temp >>  8) & 0xF] & 0xF000) ^
			(T4_SML[(temp >>  4) & 0xF] & 0x0F00) ^
			(T4_SML[(temp >>  0) & 0xF] & 0x00F0) ^
			(T4_SML[(temp >> 12) & 0xF] & 0x000F) ^
			RCON_SML[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk2 ^ rk3;
		key = (rk0 << 48) ^ (rk1 << 32) ^ (rk2 << 16) ^ rk3;
		printf("key[%02d]                       : %016llx\n", roundCount + 1, key);
		roundKeys64[roundCount + 1] = key;
	}
	printf("-------------------------------\n");
}

__global__ void smallAesOneTable(u64* roundKeys, u16* t0G, u16* t4G, u32* range, u32* prob) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	// <SHARED MEMORY>
	__shared__ u16 t0S[16], t4S[16];
	__shared__ u64 rkS[11];
	if (threadIdx.x < 11) {
		rkS[threadIdx.x] = roundKeys[threadIdx.x];
	}
	if (threadIdx.x < 16) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];  // Sbox
	}
	// </SHARED MEMORY>
	__syncthreads();

	u64 ptInit, state, temp;
	u32 threadRange = *range;
	u64 threadRangeStart = 0x0000000000000000U + threadIndex * threadRange * 0x0000000100000001U;
	ptInit = threadRangeStart;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		//if (threadIndex == 0) {
		//	printf("Plaintext: %016llx\n", ptInit);
		//}

		state = ptInit;

		// First round just XORs input with key.
		state = state ^ rkS[0];
		//if (threadIndex == 0) {
		//	printf("1 key xor: %016llx\n", state);
		//}

		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			temp = 0;
			temp ^= t0S[(state >> 60) & 0xF] ^ ROTL16(t0S[(state >> 40) & 0xF], 4) ^ ROTL16(t0S[(state >> 20) & 0xF], 8) ^ ROTL16(t0S[(state) & 0xF], 12);
			//if (threadIndex == 0) {
			//	printf("roundCount: %d temp: %016llx\n", roundCount, temp);
			//}
			temp = temp << 16;
			temp ^= t0S[(state >> 44) & 0xF] ^ ROTL16(t0S[(state >> 24) & 0xF], 4) ^ ROTL16(t0S[(state >> 4) & 0xF], 8) ^ ROTL16(t0S[(state >> 48) & 0xF], 12);
			//if (threadIndex == 0) {
			//	printf("roundCount: %d temp: %016llx\n", roundCount, temp);
			//}
			temp = temp << 16;
			temp ^= t0S[(state >> 28) & 0xF] ^ ROTL16(t0S[(state >> 8) & 0xF], 4) ^ ROTL16(t0S[(state >> 52) & 0xF], 8) ^ ROTL16(t0S[(state >> 32) & 0xF], 12);
			//if (threadIndex == 0) {
			//	printf("roundCount: %d temp: %016llx\n", roundCount, temp);
			//}
			temp = temp << 16;
			temp ^= t0S[(state >> 12) & 0xF] ^ ROTL16(t0S[(state >> 56) & 0xF], 4) ^ ROTL16(t0S[(state >> 36) & 0xF], 8) ^ ROTL16(t0S[(state >> 16) & 0xF], 12);
			//if (threadIndex == 0) {
			//	printf("roundCount: %d temp: %016llx\n", roundCount, temp);
			//}
			state = temp ^ rkS[roundCount + 1];
			//if (threadIndex == 0) {
			//	printf("roundCount: %d temp: %016llx [KEY]\n", roundCount, state);
			//}

			// Probability calculation
			#ifdef PROB_1
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 48) & 0xF], 1);
			}
			#endif
			#ifdef PROB_2
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 56) & 0xFF], 1);
			}
			#endif
		}
		// Last round uses s-box directly and XORs to produce output.

		temp = 0;
		temp ^= (t4S[(state >> 60) & 0xf] & 0xF000) ^ (t4S[(state >> 40) & 0xf] & 0x0F00) ^ (t4S[(state >> 20) & 0xf] & 0x00F0) ^ (t4S[(state) & 0xF] & 0x000F);
		//if (threadIndex == 0) {
		//	printf("roundCount: %d temp: %016llx\n", 9, temp);
		//}
		temp = temp << 16;
		temp ^= (t4S[(state >> 44) & 0xf] & 0xF000) ^ (t4S[(state >> 24) & 0xf] & 0x0F00) ^ (t4S[(state >> 4) & 0xf] & 0x00F0) ^ (t4S[(state >> 48) & 0xF] & 0x000F);
		//if (threadIndex == 0) {
		//	printf("roundCount: %d temp: %016llx\n", 9, temp);
		//}
		temp = temp << 16;
		temp ^= (t4S[(state >> 28) & 0xf] & 0xF000) ^ (t4S[(state >> 8) & 0xf] & 0x0F00) ^ (t4S[(state >> 52) & 0xf] & 0x00F0) ^ (t4S[(state >> 32) & 0xF] & 0x000F);
		//if (threadIndex == 0) {
		//	printf("roundCount: %d temp: %016llx\n", 9, temp);
		//}
		temp = temp << 16;
		temp ^= (t4S[(state >> 12) & 0xf] & 0xF000) ^ (t4S[(state >> 56) & 0xf] & 0x0F00) ^ (t4S[(state >> 36) & 0xf] & 0x00F0) ^ (t4S[(state >> 16) & 0xF] & 0x000F);
		//if (threadIndex == 0) {
		//	printf("roundCount: %d temp: %016llx\n", 9, temp);
		//}
		state = temp ^ rkS[ROUND_COUNT];
		//if (threadIndex == 0) {
		//	printf("roundCount: %d temp: %016llx [KEY]\n", 9, state);
		//}

		//printf("Thread: %d rangeCount: %d ciphertext: %016llx\n", threadIndex, rangeCount, state);

		ptInit += 0x0000000100000001U;

		//atomicAdd(&totalEncryptionsG, 1);
		//atomicXor(&ciphertextResultG, state);
	}

	if (state == 0x54366115edc783e1) {
		printf("*****************\n");
	}
}
__global__ void smallAesOneTableExtendedSharedMemory(u64* roundKeys, u16* t0G, u16* t4G, u32* range, u32* prob) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	// <SHARED MEMORY>
	__shared__ u16 t0S[16][SHARED_MEM_BANK_SIZE], t4S[16][SHARED_MEM_BANK_SIZE];
	__shared__ u64 rkS[11][SHARED_MEM_BANK_SIZE];
	if (threadIdx.x < 11) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			rkS[threadIdx.x][bankIndex] = roundKeys[threadIdx.x];
		}
	}
	if (threadIdx.x < 16) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];  // Sbox
		}
	}
	// </SHARED MEMORY>
	__syncthreads();

	u64 ptInit, state, temp;
	u32 threadRange = *range;
	u64 threadRangeStart = 0x0000000000000000U + threadIndex * threadRange * 0x0000000100000001U;
	ptInit = threadRangeStart;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		state = ptInit;

		// First round just XORs input with key.
		state = state ^ rkS[0][warpThreadIndex];
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			temp = 0;
			temp ^= t0S[(state >> 60) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 40) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 20) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state) & 0xF][warpThreadIndex], 12);
			temp = temp << 16;
			temp ^= t0S[(state >> 44) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 24) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 4) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state >> 48) & 0xF][warpThreadIndex], 12);
			temp = temp << 16;
			temp ^= t0S[(state >> 28) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 8) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 52) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state >> 32) & 0xF][warpThreadIndex], 12);
			temp = temp << 16;
			temp ^= t0S[(state >> 12) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 56) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 36) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state >> 16) & 0xF][warpThreadIndex], 12);
			state = temp ^ rkS[roundCount + 1][warpThreadIndex];

			// Probability calculation
			#ifdef PROB_1
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 48) & 0xF], 1);
			}
			#endif
			#ifdef PROB_2
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 56) & 0xFF], 1);
			}
			#endif
		}
		// Last round uses s-box directly and XORs to produce output.

		temp = 0;
		temp ^= (t4S[(state >> 60) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 40) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 20) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state) & 0xF][warpThreadIndex] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 44) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 24) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 4) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state >> 48) & 0xF][warpThreadIndex] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 28) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 8) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 52) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state >> 32) & 0xF][warpThreadIndex] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 12) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 56) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 36) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state >> 16) & 0xF][warpThreadIndex] & 0x000F);
		state = temp ^ rkS[ROUND_COUNT][warpThreadIndex];

		ptInit += 0x0000000100000001U;

		//atomicAdd(&totalEncryptionsG, 1);
		//atomicXor(&ciphertextResultG, state);
	}

	if (state == 0x54366115edc783e1) {
		printf("*****************\n");
	}
}
__global__ void smallAesOneTableExtendedSharedMemoryOnlyTable(u64* roundKeys, u16* t0G, u16* t4G, u32* range, u32* prob) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	// <SHARED MEMORY>
	__shared__ u16 t0S[16][SHARED_MEM_BANK_SIZE], t4S[16];
	__shared__ u64 rkS[11];
	if (threadIdx.x < 11) {
		rkS[threadIdx.x] = roundKeys[threadIdx.x];
	}
	if (threadIdx.x < 16) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}
		t4S[threadIdx.x] = t4G[threadIdx.x];  // Sbox
	}
	// </SHARED MEMORY>
	__syncthreads();

	u64 ptInit, state, temp;
	u32 threadRange = *range;
	u64 threadRangeStart = 0x0000000000000000U + threadIndex * threadRange * 0x0000000100000001U;
	ptInit = threadRangeStart;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		state = ptInit;

		// First round just XORs input with key.
		state = state ^ rkS[0];
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			temp = 0;
			temp ^= t0S[(state >> 60) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 40) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 20) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state) & 0xF][warpThreadIndex], 12);
			temp = temp << 16;
			temp ^= t0S[(state >> 44) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 24) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 4) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state >> 48) & 0xF][warpThreadIndex], 12);
			temp = temp << 16;
			temp ^= t0S[(state >> 28) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 8) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 52) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state >> 32) & 0xF][warpThreadIndex], 12);
			temp = temp << 16;
			temp ^= t0S[(state >> 12) & 0xF][warpThreadIndex] ^ ROTL16(t0S[(state >> 56) & 0xF][warpThreadIndex], 4) ^ ROTL16(t0S[(state >> 36) & 0xF][warpThreadIndex], 8) ^ ROTL16(t0S[(state >> 16) & 0xF][warpThreadIndex], 12);
			state = temp ^ rkS[roundCount + 1];

			// Probability calculation
			#ifdef PROB_1
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 48) & 0xF], 1);
			}
			#endif
			#ifdef PROB_2
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 56) & 0xFF], 1);
			}
			#endif
		}
		// Last round uses s-box directly and XORs to produce output.

		temp = 0;
		temp ^= (t4S[(state >> 60) & 0xf] & 0xF000) ^ (t4S[(state >> 40) & 0xf] & 0x0F00) ^ (t4S[(state >> 20) & 0xf] & 0x00F0) ^ (t4S[(state) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 44) & 0xf] & 0xF000) ^ (t4S[(state >> 24) & 0xf] & 0x0F00) ^ (t4S[(state >> 4) & 0xf] & 0x00F0) ^ (t4S[(state >> 48) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 28) & 0xf] & 0xF000) ^ (t4S[(state >> 8) & 0xf] & 0x0F00) ^ (t4S[(state >> 52) & 0xf] & 0x00F0) ^ (t4S[(state >> 32) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 12) & 0xf] & 0xF000) ^ (t4S[(state >> 56) & 0xf] & 0x0F00) ^ (t4S[(state >> 36) & 0xf] & 0x00F0) ^ (t4S[(state >> 16) & 0xF] & 0x000F);
		state = temp ^ rkS[ROUND_COUNT];

		ptInit += 0x0000000100000001U;

		//atomicAdd(&totalEncryptionsG, 1);
		//atomicXor(&ciphertextResultG, state);
	}

	if (state == 0x54366115edc783e1) {
		printf("*****************\n");
	}
}
__global__ void smallAesFourTable(u64* roundKeys, u16* t0G, u16* t1G, u16* t2G, u16* t3G, u16* t4G, u32* range, u32* prob) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	// <SHARED MEMORY>
	__shared__ u16 t0S[16], t1S[16], t2S[16], t3S[16], t4S[16];
	__shared__ u64 rkS[11];
	if (threadIdx.x < 11) {
		rkS[threadIdx.x] = roundKeys[threadIdx.x];
	}
	if (threadIdx.x < 16) {
		t0S[threadIdx.x] = t0G[threadIdx.x];
		t1S[threadIdx.x] = t1G[threadIdx.x];
		t2S[threadIdx.x] = t2G[threadIdx.x];
		t3S[threadIdx.x] = t3G[threadIdx.x];
		t4S[threadIdx.x] = t4G[threadIdx.x];  // Sbox
	}
	// </SHARED MEMORY>
	__syncthreads();

	u64 ptInit, state, temp;
	u32 threadRange = *range;
	u64 threadRangeStart = 0x0000000000000000U + threadIndex * threadRange * 0x0000000100000001U;
	ptInit = threadRangeStart;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		state = ptInit;

		// First round just XORs input with key.
		state = state ^ rkS[0];
		for (u8 roundCount = 0; roundCount < ROUND_5; roundCount++) {
			// Table based round function
			temp = 0;
			temp ^= t0S[(state >> 60) & 0xF] ^ t1S[(state >> 40) & 0xF] ^ t2S[(state >> 20) & 0xF] ^ t3S[(state) & 0xF];
			temp = temp << 16;
			temp ^= t0S[(state >> 44) & 0xF] ^ t1S[(state >> 24) & 0xF] ^ t2S[(state >> 4) & 0xF] ^ t3S[(state >> 48) & 0xF];
			temp = temp << 16;
			temp ^= t0S[(state >> 28) & 0xF] ^ t1S[(state >> 8) & 0xF] ^ t2S[(state >> 52) & 0xF] ^ t3S[(state >> 32) & 0xF];
			temp = temp << 16;
			temp ^= t0S[(state >> 12) & 0xF] ^ t1S[(state >> 56) & 0xF] ^ t2S[(state >> 36) & 0xF] ^ t3S[(state >> 16) & 0xF];
			state = temp ^ rkS[roundCount + 1];
			
		}
		// Last round uses s-box directly and XORs to produce output.

		temp = 0;
		temp ^= (t4S[(state >> 60) & 0xf] & 0xF000) ^ (t4S[(state >> 40) & 0xf] & 0x0F00) ^ (t4S[(state >> 20) & 0xf] & 0x00F0) ^ (t4S[(state) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 44) & 0xf] & 0xF000) ^ (t4S[(state >> 24) & 0xf] & 0x0F00) ^ (t4S[(state >> 4) & 0xf] & 0x00F0) ^ (t4S[(state >> 48) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 28) & 0xf] & 0xF000) ^ (t4S[(state >> 8) & 0xf] & 0x0F00) ^ (t4S[(state >> 52) & 0xf] & 0x00F0) ^ (t4S[(state >> 32) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 12) & 0xf] & 0xF000) ^ (t4S[(state >> 56) & 0xf] & 0x0F00) ^ (t4S[(state >> 36) & 0xf] & 0x00F0) ^ (t4S[(state >> 16) & 0xF] & 0x000F);
		state = temp ^ rkS[ROUND_COUNT];

		// Probability calculation
		#ifdef PROB_1
		atomicAdd(&prob[(state >> 48) & 0xF], 1);
		#endif
		#ifdef PROB_2
		atomicAdd(&prob[(state >> 56) & 0xFF], 1);
		#endif

		//if ((threadIndex == 1048575 && rangeCount == threadRange - 1) || (threadIndex == 0 && rangeCount == 0)) {
		//	printf("Thread: %d rangeCount: %d plaintext: %016llx ciphertext: %016llx\n", threadIndex, rangeCount, ptInit, state);
		//}

		ptInit += 0x0000000100000001U;

		atomicAdd(&totalEncryptionsG, 1);
		atomicXor(&ciphertextResultG, state);

	}

	if (state == 0xff3041fd1d8669ad) {
		printf("*****************\n");
	}
}
__global__ void smallAesFourTableExtendedSharedMemory(u64* roundKeys, u16* t0G, u16* t1G, u16* t2G, u16* t3G, u16* t4G, u32* range, u32* prob) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	// <SHARED MEMORY>
	__shared__ u16 t0S[16][SHARED_MEM_BANK_SIZE], t1S[16][SHARED_MEM_BANK_SIZE], t2S[16][SHARED_MEM_BANK_SIZE], t3S[16][SHARED_MEM_BANK_SIZE], t4S[16][SHARED_MEM_BANK_SIZE];
	__shared__ u64 rkS[11][SHARED_MEM_BANK_SIZE];
	if (threadIdx.x < 11) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			rkS[threadIdx.x][bankIndex] = roundKeys[threadIdx.x];
		}
	}
	if (threadIdx.x < 16) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			t1S[threadIdx.x][bankIndex] = t1G[threadIdx.x];
			t2S[threadIdx.x][bankIndex] = t2G[threadIdx.x];
			t3S[threadIdx.x][bankIndex] = t3G[threadIdx.x];
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];  // Sbox
		}
	}
	// </SHARED MEMORY>
	__syncthreads();

	u64 ptInit, state, temp;
	u32 threadRange = *range;
	u64 threadRangeStart = 0x0000000000000000U + threadIndex * threadRange * 0x0000000100000001U;
	ptInit = threadRangeStart;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		state = ptInit;

		// First round just XORs input with key.
		state = state ^ rkS[0][warpThreadIndex];
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			temp = 0;
			temp ^= t0S[(state >> 60) & 0xF][warpThreadIndex] ^ t1S[(state >> 40) & 0xF][warpThreadIndex] ^ t2S[(state >> 20) & 0xF][warpThreadIndex] ^ t3S[(state) & 0xF][warpThreadIndex];
			temp = temp << 16;
			temp ^= t0S[(state >> 44) & 0xF][warpThreadIndex] ^ t1S[(state >> 24) & 0xF][warpThreadIndex] ^ t2S[(state >> 4) & 0xF][warpThreadIndex] ^ t3S[(state >> 48) & 0xF][warpThreadIndex];
			temp = temp << 16;
			temp ^= t0S[(state >> 28) & 0xF][warpThreadIndex] ^ t1S[(state >> 8) & 0xF][warpThreadIndex] ^ t2S[(state >> 52) & 0xF][warpThreadIndex] ^ t3S[(state >> 32) & 0xF][warpThreadIndex];
			temp = temp << 16;
			temp ^= t0S[(state >> 12) & 0xF][warpThreadIndex] ^ t1S[(state >> 56) & 0xF][warpThreadIndex] ^ t2S[(state >> 36) & 0xF][warpThreadIndex] ^ t3S[(state >> 16) & 0xF][warpThreadIndex];
			state = temp ^ rkS[roundCount + 1][warpThreadIndex];

			// Probability calculation
			#ifdef PROB_1
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 48) & 0xF], 1);
			}
			#endif
			#ifdef PROB_2
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 56) & 0xFF], 1);
			}
			#endif
		}
		// Last round uses s-box directly and XORs to produce output.

		temp = 0;
		temp ^= (t4S[(state >> 60) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 40) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 20) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state) & 0xF][warpThreadIndex] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 44) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 24) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 4) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state >> 48) & 0xF][warpThreadIndex] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 28) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 8) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 52) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state >> 32) & 0xF][warpThreadIndex] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 12) & 0xf][warpThreadIndex] & 0xF000) ^ (t4S[(state >> 56) & 0xf][warpThreadIndex] & 0x0F00) ^ (t4S[(state >> 36) & 0xf][warpThreadIndex] & 0x00F0) ^ (t4S[(state >> 16) & 0xF][warpThreadIndex] & 0x000F);
		state = temp ^ rkS[ROUND_COUNT][warpThreadIndex];

		ptInit += 0x0000000100000001U;

		//atomicAdd(&totalEncryptionsG, 1);
		//atomicXor(&ciphertextResultG, state);
	}

	if (state == 0x54366115edc783e1) {
		printf("*****************\n");
	}
}

__global__ void smallAesFourTableExtendedSharedMemoryOnlyTables(u64* roundKeys, u16* t0G, u16* t1G, u16* t2G, u16* t3G, u16* t4G, u32* range, u32* prob) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	// <SHARED MEMORY>
	__shared__ u16 t0S[16][SHARED_MEM_BANK_SIZE], t1S[16][SHARED_MEM_BANK_SIZE], t2S[16][SHARED_MEM_BANK_SIZE], t3S[16][SHARED_MEM_BANK_SIZE], t4S[16];
	__shared__ u64 rkS[11];
	if (threadIdx.x < 11) {
		rkS[threadIdx.x] = roundKeys[threadIdx.x];
	}
	if (threadIdx.x < 16) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			t1S[threadIdx.x][bankIndex] = t1G[threadIdx.x];
			t2S[threadIdx.x][bankIndex] = t2G[threadIdx.x];
			t3S[threadIdx.x][bankIndex] = t3G[threadIdx.x];
		}
		t4S[threadIdx.x] = t4G[threadIdx.x];  // Sbox
	}
	// </SHARED MEMORY>
	__syncthreads();

	u64 ptInit, state, temp;
	u32 threadRange = *range;
	u64 threadRangeStart = 0x0000000000000000U + threadIndex * threadRange * 0x0000000100000001U;
	ptInit = threadRangeStart;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		state = ptInit;

		// First round just XORs input with key.
		state = state ^ rkS[0];
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			temp = 0;
			temp ^= t0S[(state >> 60) & 0xF][warpThreadIndex] ^ t1S[(state >> 40) & 0xF][warpThreadIndex] ^ t2S[(state >> 20) & 0xF][warpThreadIndex] ^ t3S[(state) & 0xF][warpThreadIndex];
			temp = temp << 16;
			temp ^= t0S[(state >> 44) & 0xF][warpThreadIndex] ^ t1S[(state >> 24) & 0xF][warpThreadIndex] ^ t2S[(state >> 4) & 0xF][warpThreadIndex] ^ t3S[(state >> 48) & 0xF][warpThreadIndex];
			temp = temp << 16;
			temp ^= t0S[(state >> 28) & 0xF][warpThreadIndex] ^ t1S[(state >> 8) & 0xF][warpThreadIndex] ^ t2S[(state >> 52) & 0xF][warpThreadIndex] ^ t3S[(state >> 32) & 0xF][warpThreadIndex];
			temp = temp << 16;
			temp ^= t0S[(state >> 12) & 0xF][warpThreadIndex] ^ t1S[(state >> 56) & 0xF][warpThreadIndex] ^ t2S[(state >> 36) & 0xF][warpThreadIndex] ^ t3S[(state >> 16) & 0xF][warpThreadIndex];
			state = temp ^ rkS[roundCount + 1];

			// Probability calculation
			#ifdef PROB_1
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 48) & 0xF], 1);
			}
			#endif
			#ifdef PROB_2
			if (roundCount == ROUND_5) {
				atomicAdd(&prob[(state >> 56) & 0xFF], 1);
			}
			#endif
		}
		// Last round uses s-box directly and XORs to produce output.

		temp = 0;
		temp ^= (t4S[(state >> 60) & 0xf] & 0xF000) ^ (t4S[(state >> 40) & 0xf] & 0x0F00) ^ (t4S[(state >> 20) & 0xf] & 0x00F0) ^ (t4S[(state) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 44) & 0xf] & 0xF000) ^ (t4S[(state >> 24) & 0xf] & 0x0F00) ^ (t4S[(state >> 4) & 0xf] & 0x00F0) ^ (t4S[(state >> 48) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 28) & 0xf] & 0xF000) ^ (t4S[(state >> 8) & 0xf] & 0x0F00) ^ (t4S[(state >> 52) & 0xf] & 0x00F0) ^ (t4S[(state >> 32) & 0xF] & 0x000F);
		temp = temp << 16;
		temp ^= (t4S[(state >> 12) & 0xf] & 0xF000) ^ (t4S[(state >> 56) & 0xf] & 0x0F00) ^ (t4S[(state >> 36) & 0xf] & 0x00F0) ^ (t4S[(state >> 16) & 0xF] & 0x000F);
		state = temp ^ rkS[ROUND_COUNT];

		ptInit += 0x0000000100000001U;

		//atomicAdd(&totalEncryptionsG, 1);
		//atomicXor(&ciphertextResultG, state);
	}

	if (state == 0x54366115edc783e1) {
		printf("*****************\n");
	}
}

__host__ int mainSmall() {
	printf("\n");
	printf("########## Small AES Implementation ##########\n");
	printf("\n");

	// Initialize round keys
	u64 *rk64, *roundKeys64;
	gpuErrorCheck(cudaMallocManaged(&rk64, 1 * sizeof(u64)));
	gpuErrorCheck(cudaMallocManaged(&roundKeys64, 11 * sizeof(u64)));
	*rk64 = 0x12ab12ab12ab12abU;

	// Initialize tables
	u16 *t0Sml, *t1Sml, *t2Sml, *t3Sml, *t4Sml;
	gpuErrorCheck(cudaMallocManaged(&t0Sml, 16 * sizeof(u16)));
	gpuErrorCheck(cudaMallocManaged(&t1Sml, 16 * sizeof(u16)));
	gpuErrorCheck(cudaMallocManaged(&t2Sml, 16 * sizeof(u16)));
	gpuErrorCheck(cudaMallocManaged(&t3Sml, 16 * sizeof(u16)));
	gpuErrorCheck(cudaMallocManaged(&t4Sml, 16 * sizeof(u16)));
	for (int i = 0; i < 16; i++) {
		t0Sml[i] = T0_SML[i];
		t1Sml[i] = T1_SML[i];
		t2Sml[i] = T2_SML[i];
		t3Sml[i] = T3_SML[i];
		t4Sml[i] = T4_SML[i];
	}

	// Initialize RCON values
	u32* rconSml;
	gpuErrorCheck(cudaMallocManaged(&rconSml, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rconSml[i] = RCON_SML[i];
	}

	// Initialize probability values
	u32* prob;
	#if defined(PROB_1)
	gpuErrorCheck(cudaMallocManaged(&prob, PROB_SIZE_1 * sizeof(u32)));
	for (int i = 0; i < PROB_SIZE_1; i++) {
		prob[i] = 0;
	}
	#endif
	#if defined(PROB_2)
	gpuErrorCheck(cudaMallocManaged(&prob, PROB_SIZE_2 * sizeof(u32)));
	for (int i = 0; i < PROB_SIZE_2; i++) {
		prob[i] = 0;
	}
	#endif

	// Calculate range
	printf("-------------------------------\n");
	u32* range = calculateRange();
	printf("Initial Key                   : %016llx\n", *rk64);
	printf("Initial Plaintext             : %016llx\n", 0);
	printf("-------------------------------\n");

	// Key Expansion
	keyExpansion(*rk64, roundKeys64);

	clock_t beginTime = clock();
	// Kernels
	//smallAesOneTable<<<BLOCKS, THREADS>>>(roundKeys64, t0Sml, t4Sml, range, prob);
	//smallAesOneTableExtendedSharedMemory<<<BLOCKS, THREADS>>>(roundKeys64, t0Sml, t4Sml, range, prob);
	//smallAesOneTableExtendedSharedMemoryOnlyTable<<<BLOCKS, THREADS>>>(roundKeys64, t0Sml, t4Sml, range, prob);
	// fastest
	smallAesFourTable<<<BLOCKS, THREADS>>>(roundKeys64, t0Sml, t1Sml, t2Sml, t3Sml, t4Sml, range, prob);
	//smallAesFourTableExtendedSharedMemory<<<BLOCKS, THREADS>>>(roundKeys64, t0Sml, t1Sml, t2Sml, t3Sml, t4Sml, range, prob);
	//smallAesFourTableExtendedSharedMemoryOnlyTables<<<BLOCKS, THREADS>>>(roundKeys64, t0Sml, t1Sml, t2Sml, t3Sml, t4Sml, range, prob);

	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();

	u64 totEncryption;
	cudaMemcpyFromSymbol(&totEncryption, totalEncryptionsG, sizeof(u64));
	printf("Total encryptions : %I64d\n", totEncryption);
	u64 ctResult;
	cudaMemcpyFromSymbol(&ctResult, ciphertextResultG, sizeof(u64));
	printf("Ciphertext result : %I64d %016llx\n", ctResult, ctResult);
	printf("-------------------------------\n");

	#if defined(PROB_1)
	u64 totalProb = 0;
	printf("Probability(1):\n");
	for (int i = 0; i < PROB_SIZE_1; i++) {
		printf("%x : %d\n", i, prob[i]);
		totalProb += prob[i];
	}
	printf("Total prob count: %I64d\n", totalProb, totalProb);
	printf("-------------------------------\n");
	#endif
	#if defined(PROB_2)
	u64 totalProb = 0;
	printf("Probability(2):\n");
	for (int i = 0; i < PROB_SIZE_2; i++) {
		printf("%x : %d\n", i, prob[i]);
		totalProb += prob[i];
	}
	printf("Total prob count: %I64d\n", totalProb);
	printf("-------------------------------\n");
	#endif

	// Free alocated arrays
	cudaFree(range);
	cudaFree(rk64);
	cudaFree(roundKeys64);
	cudaFree(prob);
	cudaFree(t0Sml);
	cudaFree(t1Sml);
	cudaFree(t2Sml);
	cudaFree(t3Sml);
	cudaFree(t4Sml);
	cudaFree(rconSml);

	return 0;
}

