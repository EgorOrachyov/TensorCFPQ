#ifndef COMPUTATION_MODULES_H
#define COMPUTATION_MODULES_H

#include "kernels.h"
#include "context.h"

#define _TRUE_  1
#define _FALSE_ 0

///
///
int cfpqTensorRun(Automata* a, Graph* g, Tensor* t) {
    // debug
    // int buffer[1024];

    // Setup atomic bool flag to check, whether result changes
    int* flag = NULL;
    createFlag(&flag);

    // Check, whether g.current != g.next
    int graphChanges = _TRUE_;

    // Width for int packed bool matrix
    int W = (int)(g->N / _BITS_COUNT_ + (g->N % _BITS_COUNT_ ? 1 : 0));
    int N = g->N;

    dim3 grid(W, N, 1);
    dim3 gridForGraph(N, N, 1);
    dim3 threads(1, 1, 1);
    dim3 threadsPerBits(_BITS_COUNT_, 1, 1);

    while (graphChanges) {

        graphChanges = _FALSE_;

        // Tensor product a.mat x graph.current
        // Store result in mat t.current
        {
            int** tensor = t->mat;

            for (int i = 0; i < t->N; i++) {
                for (int j = 0; j < t->N; j++) {

                    int value = a->mat[i * a->N + j];

                    if (value == 0) {
                        continue;
                    }

                    int* block = tensor[i * t->N + j];
                    int hasBlock = block != NULL;
                    if (!hasBlock) {
                        block = t->tmpBlock;
                    }

                    resetFlag(flag);
                    kernel_numMatProductPacked <<< grid, threads >>> (value, N, g->current, W, block, flag);
                    checkCudaError(cudaDeviceSynchronize(), "SYNC");

                    if (!hasBlock && getFlagValue(flag)) {
                        // Save new block in tensor
                        tensor[i * t->N + j] = block;

                        // If we used tmp block, than we need to allocate another block
                        allocateTmpBlock(t);
                    }

                    // debug
//                        printf("Block[%i,%i]\n", i, j);
//                        gpu_copyDeviceToHost(buffer, block, t->tmpBlockSize);
//                        printMatrixBoolPacked("a & M block", N, W, buffer);
                }
            }
        }

        // Transitive clojure for tensor
        // packed boolean matrix while matrix changes
        {
            int tensorChanges = _TRUE_;

            // Save here results of the clojure
            int** tensor = t->mat;

            while (tensorChanges) {

                tensorChanges = _FALSE_;

                // Compute d[i][j] = Sum[k=0..N-1] a[i][k]*b[k][j] + c[i][j]
                // There could be some cases:
                // 1. c[i][j] == null
                // 2. Sum[k=0..N-1] a[i][k]*b[k][j] == null

                for (int i = 0; i < t->N; i++) {
                    for (int j = 0; j < t->N; j++) {

                        int* d = t->tmpBlock;

                        int mulAtOnce = _FALSE_;

                        for (int k = 0; k < t->N; k++) {

                            int* a = tensor[i * t->N + k];
                            int* b = tensor[k * t->N + j];

                            if ((a == NULL) || (b == NULL)) {
                                continue;
                            }

                            kernel_boolMatMullPacked <<< grid, threadsPerBits >>> (N, W, a, b, d);
                            checkCudaError(cudaDeviceSynchronize(), "SYNC");
                            mulAtOnce = _TRUE_;
                        }

                        if (mulAtOnce) {
                            int* c = tensor[i * t->N + j];

                            // Has no block c from prev iteration
                            if (c == NULL) {
                                resetFlag(flag);
                                kernel_boolMatCheckZero <<< grid, threads >>> (W, d, flag);
                                checkCudaError(cudaDeviceSynchronize(), "SYNC");

                                // Sum[k=0..N-1] a[i][k] * b[k][j] not zero
                                if (getFlagValue(flag)) {
                                    tensor[i * t->N + j] = d;

                                    // If we use tmp block as d, we need to allocate another tmp
                                    allocateTmpBlock(t);

                                    // Tensor changes, because d != null, c == null
                                    tensorChanges = _TRUE_;
                                }
                            }
                            // Has block from prev iteration, need: d |= c and check: c != d
                            else {
                                resetFlag(flag);
                                kernel_boolMatOrPacked <<< grid, threads >>> (W, d, c, d, c, flag);
                                checkCudaError(cudaDeviceSynchronize(), "SYNC");

                                // Flag true, if block d not equals c from prev iteration
                                if (getFlagValue(flag)) {
                                    tensorChanges = _TRUE_;
                                }

                                tensor[i * t->N + j] = d;

                                // Return prev iteration c block as new tmp
                                t->tmpBlock = c;
                                gpu_memorySet(c, 0x0, t->tmpBlockSize);
                            }
                        }
                        else {
                            // Sum[k=0..N-1] a[i][k] * b[k][j] equals zero
                            // Block d := c block from prev iteration
                        }
                    }
                }
            }

            // debug
//            printf("Tensor clojure\n");
//            for (int i = 0; i < t->N; i++) {
//                for (int j = 0; j < t->N; j++) {
//                    if (tensor[i * t->N + j] != NULL) {
//                        printf("[%i,%i]", i, j);
//                        gpu_copyDeviceToHost(buffer, tensor[i * t->N + j], t->tmpBlockSize);
//                        printMatrixBoolPacked("", N, W, buffer);
//                    }
//                    else {
//                        printf("[%i,%i] Null\n", i, j);
//                    }
//                }
//            }
        }

        // Update graph matrix
        // Accordingly to automata state mat add new edges in graph matrix
        {
            int* current = g->current;
            int* next = g->next;

            for (int i = 0; i < a->N; i++) {
                for (int j = 0; j < a->N; j++) {

                    int s = a->states[i * a->N + j];
                    int* block = t->mat[i * t->N + j];

                    // debug
                    // printf("[%i,%i] %i %p\n", i, j, s, block);

                    if ((s != 0) && (block != NULL)) {
                        kernel_boolMatCheckValue <<< grid, threadsPerBits >>> (s, W, block, N, next);
                        checkCudaError(cudaDeviceSynchronize(), "SYNC");
                    }
                }
            }

            resetFlag(flag);
            kernel_boolMatOrPacked <<<gridForGraph, threads>>> (N, current, next, next, current, flag);
            checkCudaError(cudaDeviceSynchronize(), "SYNC");

            if (getFlagValue(flag)) {
                graphChanges = _TRUE_;
            }

            g->next = current;
            g->current = next;

            // debug
//            gpu_copyDeviceToHost(buffer, g->current, N * N * sizeof(int));
//            printMatrix("Graph matrix", N, buffer);
        }
    }

    // debug
//    printf("Total tmp blocks allocated: %li\n", t->tmpBlocksCount);

    destroyFlag(flag);

    return 0;
}

#undef _TRUE_
#undef _FALSE_

#endif // COMPUTATION_MODULES_H