#ifndef COMPUTATION_MISC_H
#define COMPUTATION_MISC_H

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cassert>

/// Checks, whether there is some error
/// Common pattern: checkCudaErr(someCudaFunction(...), someMessage);
inline void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    assert(true);
  }
}


/// Print an arbitrary matrix of size NxN
inline void printMatrix(const char* name, int N, int* m) {
    printf("Matrix: %s\n", name);
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            printf("%i ", m[row * N + col]);
        }
        printf("\n");
    }
    printf("\n");
}


/// Print an arbitrary bool matrix of size NxN
inline void printMatrixBool(const char* name, int N, int* m) {
	printf("Matrix: %s\n", name);
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            printf("%i", m[row * N + col] != 0);
        }
        printf("\n");
    }
    printf("\n");
}

/// Print an arbitrary packed bool matrix of size NxN (byte size NxW)
inline void printMatrixBoolPacked(const char* name, int N, int W, int* m) {
    const int bits = sizeof(int) * 8;
    printf("Matrix: %s\n", name);
    for (int row = 0; row < N; row++) {
        for (int word = 0; word < W; word++) {
            for (int t = 0; t < bits; t++) {
                int j = word * bits + t;

                if (j < N) {
                    printf("%i", ((m[row * W + word] >> (bits - 1 - t)) & 0x1) != 0);
                } else {
                    break;
                }
            }

        }
        printf("\n");
    }
    printf("\n");
}

#endif // COMPUTATION_MISC_H