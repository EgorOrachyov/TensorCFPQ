#ifndef COMPUTATION_KERNELS_H
#define COMPUTATION_KERNELS_H

#include "context.h"

#define _ATOMIC_TRUE_ 0x1
#define _BITS_COUNT_  (sizeof(int) * 8)

/// Multiply int a value to matrix b.
/// If result not zero, set flag in 0x1
/// @note Must be run with block dim (1, 1, 1)
/// @note Must be run with grid dim (mat C height, mat C packed width, 1)
__global__ void kernel_numMatProductPacked(int a, int Nb, const int* b, int Wc, int* c, int* flag) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    // int t = 0; always only one thread in block

    int acc = 0;

    #pragma unroll
    for (int t = 0; t < _BITS_COUNT_; t++) {
        int j_b = j * _BITS_COUNT_ + t;

        if (j_b < Nb) {
            int value = b[i * Nb + j_b];
                value = ((a & value) != 0 ? 0x1 : 0x0);

            acc |= (value << (_BITS_COUNT_ - 1 - t));
        }
    }

    // Accumulate data (save paths from prev iteration)
    c[i * Wc + j] |= acc;

    if (acc != 0) {
        atomicOr(flag, _ATOMIC_TRUE_);
    }
}

/// Multiply packed bool matrices a x b and add result to mat c:  c |= a x b
/// @note Must be run with block dim (_BITS_COUNT_, 1, 1)
/// @note Must be run with grid dim (N, W, 1), where N (height of the a, b, c) and W (width of the a, b, c in words)
__global__ void kernel_boolMatMullPacked(int N, int W, const int* a, const int* b, int *c) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    int t = threadIdx.x;

    __shared__ int values[_BITS_COUNT_];

    /* Each thread set own local value to 0 */
    values[t] = 0;

    int bounds = j * _BITS_COUNT_ + t;

    if (bounds < N) {
        int Ai = i;
        int Bj = j;

        for (int k = 0; k < N; k++) {
            int Aj = k / _BITS_COUNT_;
            int Aoffset = k % _BITS_COUNT_;
            int _a = (a[Ai * W + Aj] >> (_BITS_COUNT_ - 1 - Aoffset)) & 0x1;

            int Bi = k;
            int Boffset = t;
            int _b = (b[Bi * W + Bj] >> (_BITS_COUNT_ - 1 - Boffset)) & 0x1;

            values[t] |= ((_a & _b != 0) ? 0x1 : 0x0);
        }
    }

    __syncthreads();

    if (t != 0) {
        return;
    }

    int acc = 0;

    /* Pack data as int with bits: 0t ..... 31t, where t - thread id  */
    #pragma unroll
    for (int i = 0; i < _BITS_COUNT_; i++) {
        acc |= (values[i] << (_BITS_COUNT_ - 1 - i));
    }

    c[i * W + j] |= acc;
}

/// Bit matrix or operation:  c = a | b
/// If c != reference, than flag will be set in true
/// @note Must be run with block dim (1, 1, 1)
/// @note Must be run with grid dim (N, W, 1), where N (height of the a, b, c) and W (width of the a, b, c in words)
__global__ void kernel_boolMatOrPacked(int W, const int *a, const int* b, int* c, const int* reference, int* flag) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    // int t = 0; always only one thread in block

    int index = i * W + j;
    int value = reference[index];

    c[index] = a[index] | b[index];

    if (c[index] != value) {
        atomicOr(flag, _ATOMIC_TRUE_);
    }
}

/// Check if mat not null (contains only zeros)
/// @note Must be run with block dim (1, 1, 1)
/// @note Must be run with grid dim (W, N, 1) where N - a height, W - a width
__global__ void kernel_boolMatCheckZero(int W, const int *a, int* flag) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    // int t = 0; always only one thread in block

    int value = a[i * W + j];

    if (value != 0) {
        atomicOr(flag, _ATOMIC_TRUE_);
    }
}

/// Set b[i][j] |= v if a[i][j] not zero
/// @note Matrices a nad b must have the same size, but a is packed (physical size is different)
/// @note Must be run with block dim (_BITS_COUNT_, 1, 1)
/// @note Must be run with grid dim (Na, Wa, 1), where Na (height of the a) and Wa (width of the a in words)
__global__ void kernel_boolMatCheckValue(int v, int Wa, const int* a, int Nb, int* b) {
    int i = blockIdx.y;
    int j = blockIdx.x;
    int t = threadIdx.x;

    int bounds = j * _BITS_COUNT_ + t;

    if (bounds < Nb) {
        int value = (a[Wa * i + j] >> (_BITS_COUNT_ - 1 - t)) & 0x1;
        b[i * Nb + j * _BITS_COUNT_ + t] |= (value != 0x0 ? v : 0x0);
    }
}

#undef _ATOMIC_TRUE_

#endif // COMPUTATION_KERNELS_H