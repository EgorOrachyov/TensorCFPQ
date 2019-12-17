///////////////////////////////////////////////////////////////////////////////////////
/// MIT License                                                                     ///
///                                                                                 ///
/// Copyright (c) 2019 Egor Orachyov                                                ///
///                                                                                 ///
/// Permission is hereby granted, free of charge, to any person obtaining a copy    ///
/// of this software and associated documentation files (the "Software"), to deal   ///
/// in the Software without restriction, including without limitation the rights    ///
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       ///
/// copies of the Software, and to permit persons to whom the Software is           ///
/// furnished to do so, subject to the following conditions:                        ///
///                                                                                 ///
/// The above copyright notice and this permission notice shall be included in all  ///
/// copies or substantial portions of the Software.                                 ///
///                                                                                 ///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      ///
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        ///
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     ///
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          ///
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   ///
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   ///
/// SOFTWARE.                                                                       ///
///////////////////////////////////////////////////////////////////////////////////////

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