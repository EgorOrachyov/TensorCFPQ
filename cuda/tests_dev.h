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

#ifndef KERNELS_TESTS_H
#define KERNELS_TESTS_H

#include "context.h"
#include "kernels.h"
#include "solution.h"
#include "automata_loader.h"

__host__ void test_kernel_numMatProductPacked(int N, cudaEvent_t start, cudaEvent_t stop) {
    int v = 1 << 12;
    int W = (int)(N / _BITS_COUNT_ + (N % _BITS_COUNT_ ? 1 : 0));

    size_t packed = N * W * sizeof(int);
    size_t size = N * N * sizeof(int);

    int *a, *b, *c, *flag, *host;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, packed);
    cudaMallocManaged(&c, packed);
    cudaMallocManaged(&flag, sizeof(int));
    host = (int*) malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = j % (1 + i);

            if (((i + j) % 16) == 0) {
                a[i * N + j] = v;
            }
        }
    }

    gpu_memorySet(c, 0x0, packed);
    gpu_memorySet(b, 0x0, packed);

    printMatrix("Input matrix", N, a);

    dim3 blocksDim(W, N, 1);
    dim3 threadsDim(_BITS_COUNT_, 1, 1);
    dim3 singleDim(1, 1, 1);

    resetFlag(flag);
    float milliseconds = 0;

    cudaEventRecord(start);
    kernel_numMatProductPacked <<< blocksDim, singleDim >>> (v, N, a, W, b, flag);
    cudaEventRecord(stop);

    checkCudaError(cudaEventSynchronize(stop), "SYNC");
    checkCudaError(cudaGetLastError(), "GPU");

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time: %f ms\n", milliseconds);
    printf("Flag: %i\n", getFlagValue(flag));

    cudaMemcpy(host, b, packed, cudaMemcpyDeviceToHost);
    printMatrixBoolPacked("Result matrix", N, W, host);

    resetFlag(flag);
    milliseconds = 0;

    cudaEventRecord(start);
    kernel_boolMatMullPacked <<< blocksDim, threadsDim >>> (N, W, b, b, c);
    cudaEventRecord(stop);

    checkCudaError(cudaEventSynchronize(stop), "SYNC");
    checkCudaError(cudaGetLastError(), "GPU");

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time: %f ms\n", milliseconds);
    printf("Flag: %i\n", getFlagValue(flag));

    cudaMemcpy(host, c, packed, cudaMemcpyDeviceToHost);
    printMatrixBoolPacked("Result matrix", N, W, host);

    resetFlag(flag);
    milliseconds = 0;

    cudaEventRecord(start);
    kernel_boolMatOrPacked <<< blocksDim, singleDim >>> (W, c, b, c, b, flag);
    cudaEventRecord(stop);

    checkCudaError(cudaEventSynchronize(stop), "SYNC");
    checkCudaError(cudaGetLastError(), "GPU");

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time: %f ms\n", milliseconds);
    printf("Flag: %i\n", getFlagValue(flag));

    cudaMemcpy(host, c, packed, cudaMemcpyDeviceToHost);
    printMatrixBoolPacked("Result matrix", N, W, host);

    resetFlag(flag);
    milliseconds = 0;

    cudaEventRecord(start);
    kernel_boolMatCheckValue <<< blocksDim, threadsDim >>> (v, W, c, N, a);
    cudaEventRecord(stop);

    checkCudaError(cudaEventSynchronize(stop), "SYNC");
    checkCudaError(cudaGetLastError(), "GPU");

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time: %f ms\n", milliseconds);
    printf("Flag: %i\n", getFlagValue(flag));

    cudaMemcpy(host, a, size, cudaMemcpyDeviceToHost);
    printMatrix("Result matrix", N, host);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(flag);
    cudaFree(host);
}

void test_basicAutomataGraph() {
    int Ma[] = {
        0, 1, 0, 0,
        0, 0, 4, 2,
        0, 0, 0, 2,
        0, 0, 0, 0

    };

    int Ms[] = {
        0, 0, 0, 4,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    int Mg[] = {
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 0, 0, 2,
        0, 0, 2, 0,
    };

    Automata automata;
    Graph graph;
    Tensor tensor;

    initAutomata(&automata, 4, Ma, Ms);
    initGraph(&graph, 4, Mg);
    initTensor(&tensor, 4, sizeof(int) * 4);

    cfpqTensorRun(&automata, &graph, &tensor);

    gpu_copyDeviceToHost(Mg, graph.current, 4 * 4 * sizeof(int));
    printMatrix("Result graph matrix", 4, Mg);

    destroyTensor(&tensor);
    destroyGraph(&graph);
    destroyAutomata(&automata);
}

void test_example2AutomataGraph() {
    int Ma[] = {
            0, 1, 0,
            0, 0, 4,
            2, 0, 0,

    };

    int Ms[] = {
            4, 0, 0,
            0, 0, 0,
            0, 0, 0,
    };

    int Mg[] = {
            4, 1, 0, 0, 0, 0, 0,
            0, 4, 2, 0, 0, 0, 0,
            0, 0, 4, 1, 0, 0, 0,
            0, 0, 0, 4, 2, 0, 0,
            0, 0, 0, 0, 4, 1, 0,
            0, 0, 0, 0, 0, 4, 2,
            0, 0, 0, 0, 0, 0, 4,
    };

    Automata automata;
    Graph graph;
    Tensor tensor;

    initAutomata(&automata, 3, Ma, Ms);
    initGraph(&graph, 7, Mg);
    initTensor(&tensor, 3, sizeof(int) * 7);

    cfpqTensorRun(&automata, &graph, &tensor);

    gpu_copyDeviceToHost(Mg, graph.current, 7 * 7 * sizeof(int));
    printMatrix("Result graph matrix", 7, Mg);

    destroyTensor(&tensor);
    destroyGraph(&graph);
    destroyAutomata(&automata);
}

void test_wineGraph() {
    GraphData graphData;
    AutomataLoader automataLoader("../cfpq/data/RDF/automata.txt");
    automataLoader.loadGraph("../cfpq/data/RDF/wine.txt", graphData);

    Automata automata;
    Graph graph;
    Tensor tensor;

    int Na = automataLoader.getStatesCount();
    int Ng = graphData.getVerticesCount();
    int Wt = (int)(Ng / _BITS_COUNT_ + (Ng % _BITS_COUNT_ ? 1 : 0));
    int St = (int)(sizeof(int) * Wt * Ng);
    int Sg = (int)(sizeof(int) * Ng * Ng);
    int* buffer = (int*) malloc(Sg);

    initAutomata(&automata, Na, automataLoader.getMatrix().data(), automataLoader.getStates().data());
    initGraph(&graph, Ng, graphData.getMatrix().data());
    initTensor(&tensor, Na, St);

    cfpqTensorRun(&automata, &graph, &tensor);
    gpu_copyDeviceToHost(buffer, graph.current, Sg);

    int label = 4;
    int count = 0;
    for (int i = 0; i < Ng * Ng; i++) {
        if (buffer[i] == label) {
            count += 1;
        }
    }

    printf("Count: %i \n", count);

    destroyTensor(&tensor);
    destroyGraph(&graph);
    destroyAutomata(&automata);
    free(buffer);
}

void test_loop() {
    const char* graphs[] = {
            "../cfpq/data/RDF/core.txt",
            //"../cfpq/data/RDF/go.txt",
            "../cfpq/data/RDF/funding.txt",
            "../cfpq/data/RDF/pizza.txt",
            "../cfpq/data/RDF/wine.txt",
    };
    int graphsCount = sizeof(graphs) / sizeof(const char*);

    AutomataLoader automataLoader("../cfpq/data/RDF/automata.txt");

    for (int i = 0; i < graphsCount; i++) {
        GraphData graphData;
        automataLoader.loadGraph(graphs[i], graphData);

        Automata automata;
        Graph graph;
        Tensor tensor;

        int Na = automataLoader.getStatesCount();
        int Ng = graphData.getVerticesCount();
        int Wt = (int)(Ng / _BITS_COUNT_ + (Ng % _BITS_COUNT_ ? 1 : 0));
        int St = (int)(sizeof(int) * Wt * Ng);
        int Sg = (int)(sizeof(int) * Ng * Ng);
        int* buffer = (int*) malloc(Sg);

        initAutomata(&automata, Na, automataLoader.getMatrix().data(), automataLoader.getStates().data());
        initGraph(&graph, Ng, graphData.getMatrix().data());
        initTensor(&tensor, Na, St);

        cfpqTensorRun(&automata, &graph, &tensor);
        gpu_copyDeviceToHost(buffer, graph.current, Sg);

        int label = 4;
        int count = 0;
        for (int t = 0; t < Ng * Ng; t++) {
            if ((buffer[t] & label) != 0) {
                count += 1;
            }
        }

        printf("Graph: %s, Count: %i \n", graphs[i], count);

        destroyTensor(&tensor);
        destroyGraph(&graph);
        destroyAutomata(&automata);
        free(buffer);
    }
}

void test_loop_worstcase() {
    const char* graphs[] = {
            "../cfpq/data/WorstCase/worstcase_64.txt",
    };
    int graphsCount = sizeof(graphs) / sizeof(const char*);

    AutomataLoader automataLoader("../cfpq/data/WorstCase/automata.txt");

    for (int i = 0; i < graphsCount; i++) {
        GraphData graphData;
        automataLoader.loadGraph(graphs[i], graphData);

        Automata automata;
        Graph graph;
        Tensor tensor;

        int Na = automataLoader.getStatesCount();
        int Ng = graphData.getVerticesCount();
        int Wt = (int)(Ng / _BITS_COUNT_ + (Ng % _BITS_COUNT_ ? 1 : 0));
        int St = (int)(sizeof(int) * Wt * Ng);
        int Sg = (int)(sizeof(int) * Ng * Ng);
        int* buffer = (int*) malloc(Sg);

        initAutomata(&automata, Na, automataLoader.getMatrix().data(), automataLoader.getStates().data());
        initGraph(&graph, Ng, graphData.getMatrix().data());
        initTensor(&tensor, Na, St);

        cfpqTensorRun(&automata, &graph, &tensor);
        gpu_copyDeviceToHost(buffer, graph.current, Sg);

        int label = 2;
        int count = 0;
        for (int t = 0; t < Ng * Ng; t++) {
            if ((buffer[t] & label) != 0) {
                count += 1;
            }
        }

        printf("Graph: %s, Count: %i \n", graphs[i], count);

        destroyTensor(&tensor);
        destroyGraph(&graph);
        destroyAutomata(&automata);
        free(buffer);
    }
}

__host__ void tests_run() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = 32;

    test_kernel_numMatProductPacked(N, start, stop);
    //test_basicAutomataGraph();
    //test_example2AutomataGraph();
    //test_wineGraph();
    test_loop();
    test_loop_worstcase();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Finish: code 0");
}

#endif // KERNELS_TESTS_H