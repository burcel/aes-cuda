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
//
#include "128-es.cuh"
#include "128-ctr.cuh"
#include "192-es.cuh"
#include "192-ctr.cuh"
#include "256-es.cuh"
#include "256-ctr.cuh"
#include "small.cuh"
#include "silent.cuh"
#include "file-encryption.cuh"

int main() {

	// AES-128 Exhaustive Search
	main128ExhaustiveSearch();

	// AES-128 Counter Mode
	//main128Ctr();

	// AES-192 Exhaustive Search
	//main192ExhaustiveSearch();

	// AES-192 Counter Mode
	//main192Ctr();

	// AES-256 Exhaustive Search
	//main256ExhaustiveSearch();

	// AES-256 Counter Mode
	//main256Ctr();

	// Small AES probability calculation
	//mainSmall();

	// Silent
	//mainSilent();

	// File Encryption
	//mainFileEncryption();

	return 0;
}
