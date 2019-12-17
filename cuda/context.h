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

#ifndef COMPUTATION_CONTEX_H
#define COMPUTATION_CONTEX_H

#include <inttypes.h>
#include "memory.h"

/// Automata presentation
/// Contains graph with transitions and
/// additional matrix to check, whether from state i
/// to state j we can derive some Non-terminal symbol
typedef struct {
        
    int N;
    int* mat;
    int* states;

} Automata;

/// Graph structure
/// Has additional matrix to check, whether it was
/// changed in time of the loop computations.
typedef struct {
    
    int N;
    int* mat1;
    int* mat2;

    int* current;
    int* next;

} Graph;

/// Describes spars tensor matrix with block structure
/// Has additional matrix to check, whether it was
/// changed in time of the loop computations.
typedef struct {

    int N;
    int** mat;

    size_t tmpBlocksCount;
    size_t tmpBlockSize;
    int* tmpBlock;

} Tensor;

// Impl section

void initAutomata(Automata* automata, int N, const int* mat, const int* states) {
    size_t size = N * N * sizeof(int);
    automata->mat = (int*) malloc(size);
    automata->states = (int*) malloc(size);
    memcpy(automata->mat, mat, size);
    memcpy(automata->states, states, size);

    automata->N = N;
}

void destroyAutomata(Automata* automata) {
    free(automata->mat);
    free(automata->states);
    memset(automata, 0x0, sizeof(Automata));
}

void initGraph(Graph* graph, int N, const int* mat) {
    size_t size = N * N * sizeof(int);
    gpu_allocate((void**) &graph->mat1, size);
    gpu_allocate((void**) &graph->mat2, size);
    gpu_copyHostToDevice(graph->mat1, (const void*) mat, size);
    gpu_copyHostToDevice(graph->mat2, (const void*) mat, size);

    graph->N = N;
    graph->current = graph->mat1;
    graph->next = graph->mat2;
}

void destroyGraph(Graph* graph) {
    gpu_free(graph->mat1);
    gpu_free(graph->mat2);
    memset(graph, 0x0, sizeof(Graph));
}

void allocateTmpBlock(Tensor* tensor) {
    gpu_allocate((void**) &tensor->tmpBlock, tensor->tmpBlockSize);
    gpu_memorySet(tensor->tmpBlock, 0x0, tensor->tmpBlockSize);
    tensor->tmpBlocksCount += 1;
}

void initTensor(Tensor *tensor, int N, size_t blockSize) {
    size_t size = N * N * sizeof(int*);
    tensor->mat = (int**) malloc(size);
    memset(tensor->mat, 0x0, size);

    tensor->N = N;
    tensor->tmpBlockSize = blockSize;
    tensor->tmpBlocksCount = 0;

    allocateTmpBlock(tensor);
}

void destroyTensor(Tensor* tensor) {
    int N = tensor->N;
    int** mat = tensor->mat;

    size_t blocks = tensor->tmpBlocksCount;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (mat[i * N + j] != NULL) {
                gpu_free(mat[i * N + j]);
                blocks -= 1;
            }
        }
    }

    if (tensor->tmpBlock != NULL) {
        gpu_free(tensor->tmpBlock);
        blocks -= 1;
    }

    free(mat);
    memset(tensor, 0x0, sizeof(Tensor));

    assert(blocks == 0);
}

void createFlag(int** flag) {
    gpu_allocate((void**) flag, sizeof(int));
}

void destroyFlag(int* flag) {
    gpu_free(flag);
}

void resetFlag(int* flag) {
    gpu_memorySet(flag, 0x0, sizeof(int));
}

int getFlagValue(int* flag) {
    int a = 0x0;
    gpu_copyDeviceToHost(&a, flag, sizeof(int));
    return a;
}

#endif // COMPUTATION_CONTEX_H