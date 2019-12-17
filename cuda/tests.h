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

#ifndef CUTLASS_TESTS_H
#define CUTLASS_TESTS_H

#include "solution.h"
#include "context.h"
#include "automata_loader.h"
#include "timer.h"

void test_singleGraph(AutomataLoader& automataLoader, const std::string &filename, int iterations) {
    GraphData graphData;
    automataLoader.loadGraph(filename, graphData);

    double time = 0.0f;
    printf("Graph: '%s'\n", filename.c_str());

    int Na = automataLoader.getStatesCount();
    int Ng = graphData.getVerticesCount();
    int Wt = (int)(Ng / _BITS_COUNT_ + (Ng % _BITS_COUNT_ ? 1 : 0));
    int St = (int)(sizeof(int) * Wt * Ng);
    int Sg = (int)(sizeof(int) * Ng * Ng);
    int* buffer = (int*) malloc(Sg);

    for (int iter = 0; iter < iterations; iter++) {
        Automata automata;
        Graph graph;
        Tensor tensor;

        initAutomata(&automata, Na, automataLoader.getMatrix().data(), automataLoader.getStates().data());
        initGraph(&graph, Ng, graphData.getMatrix().data());
        initTensor(&tensor, Na, St);

        Timer timer;
        timer.begin();
        cfpqTensorRun(&automata, &graph, &tensor);
        timer.end();

        gpu_copyDeviceToHost(buffer, graph.current, Sg);

        const int FIRST_ITERATION = 0;
        if (iter == FIRST_ITERATION) {
            for (const auto& start: automataLoader.getStartSymbols()) {
                int label = start;
                int count = 0;
                for (int t = 0; t < Ng * Ng; t++) {
                    if ((buffer[t] & label) != 0) {
                        count += 1;
                    }
                }

                printf("Symbol: %i, Count: %i\n", start, count);
            }
        }

        if (iter == FIRST_ITERATION) {
            printf("Time: ");
        }

        printf("%lf ", timer.elapsed());
        time += timer.elapsed();

        destroyTensor(&tensor);
        destroyGraph(&graph);
        destroyAutomata(&automata);
    }

    double average = time / (double)(iterations);
    printf(" Average: %lfs\n", average);

    free(buffer);
}

void test_singleAutomata(const std::string &filename, const std::string &prefix, const char* const* graphs, int count) {
    AutomataLoader loader(filename);

    printf("Automata: '%s'\n", filename.c_str());
    for (int i = 0; i < count; i++) {
        std::string filename = prefix + graphs[i];
        const int iterations = 1;
        test_singleGraph(loader, filename, iterations);
    }
}

void test_singleAutomataIterations(const std::string &filename, const std::string &prefix,
        const char* const* graphs, const int* iterations, int count) {
    AutomataLoader loader(filename);

    printf("Automata: '%s'\n", filename.c_str());
    for (int i = 0; i < count; i++) {
        std::string filename = prefix + graphs[i];
        test_singleGraph(loader, filename, iterations[i]);
    }
}

void test_worstCase() {
    std::string prefix = "../data/";
    std::string folder = "WorstCase/";

    const char* graphs[] = {
            "worstcase_64.txt",
            "worstcase_128.txt",
            "worstcase_256.txt",
//            "worstcase_512.txt",
//            "worstcase_1024.txt",
//            "worstcase_2048.txt",
    };
    const int graphsCount = (int)(sizeof(graphs) / sizeof(const char*));

    const int iterations[] = {
            10,
            10,
            2,
//            2,
//            2,
//            2
    };

    std::string folderPrefix = prefix + folder;
    std::string filename = folderPrefix + "automata.txt";
    test_singleAutomataIterations(filename, folderPrefix, graphs, iterations, graphsCount);
}

void test_rdf() {
    std::string prefix = "../data/";
    std::string folder = "RDF/";

    const char* graphs[] = {
            "atom-primitive.txt",
            "funding.txt",
            "pizza.txt",
            "biomedical-mesure-primitive.txt",
            "generations.txt",
            "core.txt",
//            "go.txt",
            "travel.txt",
//            "enzyme.txt",
            "univ-bench.txt",
            "wine.txt",
    };
    const int graphsCount = (int)(sizeof(graphs) / sizeof(const char*));

    const int iterations[] = {
            10,
            10,
            10,
            10,
            10,
            10,
//            10,
            10,
//            10,
            10,
            10
    };

    std::string folderPrefix = prefix + folder;
    std::string filename = folderPrefix + "automata.txt";
    test_singleAutomataIterations(filename, folderPrefix, graphs, iterations, graphsCount);
}

void test_fullGraph() {
    std::string prefix = "../data/";
    std::string folder = "FullGraph/";

    const char* graphs[] = {
            "fullgraph_100.txt",
            "fullgraph_200.txt",
            "fullgraph_500.txt",
            "fullgraph_1000.txt",
//            "fullgraph_2000.txt",
//            "fullgraph_5000.txt"
    };
    const int graphsCount = (int)(sizeof(graphs) / sizeof(const char*));

    const int iterations[] = {
            10,
            10,
            4,
            4,
//            2,
//            2
    };

    std::string folderPrefix = prefix + folder;
    std::string filename = folderPrefix + "automata.txt";
    test_singleAutomataIterations(filename, folderPrefix, graphs, iterations, graphsCount);
}

void test_memoryAliases() {
    std::string prefix = "../data/";
    std::string folder = "MemoryAliases/";

    const char* graphs[] = {
            "bzip2.txt",
//            "gzip.txt",
            "ls.txt",
            "pr.txt",
            "wc.txt",
    };
    const int graphsCount = (int)(sizeof(graphs) / sizeof(const char*));

    const int iterations[] = {
            10,
//            10,
            10,
            10,
            10
    };

    std::string folderPrefix = prefix + folder;
    std::string filename = folderPrefix + "automata.txt";
    test_singleAutomataIterations(filename, folderPrefix, graphs, iterations, graphsCount);
}

void test_graphOutput() {
    std::string prefix = "../data/";
    std::string folder = "MemoryAliases/";
    std::string folderPrefix = prefix + folder;
    std::string filename = folderPrefix + "automata.txt";
    std::string graphname = folderPrefix + "wc.txt";
    std::string outputname = folderPrefix + "generated.txt";

    {
        GraphData graphData;
        AutomataLoader automataLoader(filename);
        automataLoader.loadGraph(graphname, graphData);

        int Na = automataLoader.getStatesCount();
        int Ng = graphData.getVerticesCount();
        int Wt = (int)(Ng / _BITS_COUNT_ + (Ng % _BITS_COUNT_ ? 1 : 0));
        int St = (int)(sizeof(int) * Wt * Ng);
        int Sg = (int)(sizeof(int) * Ng * Ng);
        int* buffer = (int*) malloc(Sg);

        Automata automata;
        Graph graph;
        Tensor tensor;

        initAutomata(&automata, Na, automataLoader.getMatrix().data(), automataLoader.getStates().data());
        initGraph(&graph, Ng, graphData.getMatrix().data());
        initTensor(&tensor, Na, St);

        cfpqTensorRun(&automata, &graph, &tensor);
        gpu_copyDeviceToHost(buffer, graph.current, Sg);

        for (const auto& start: automataLoader.getStartSymbols()) {
            int label = start;
            int count = 0;
            for (int t = 0; t < Ng * Ng; t++) {
                if ((buffer[t] & label) != 0) {
                    count += 1;
                }
            }

            printf("Symbol: %i, Count: %i\n", start, count);
        }

        automataLoader.generateGraphVis(outputname, buffer, Ng);

        destroyTensor(&tensor);
        destroyGraph(&graph);
        destroyAutomata(&automata);

        free(buffer);
    }
}

void test_run() {
    test_worstCase();
    test_rdf();
    test_fullGraph();
    test_memoryAliases();
//    test_graphOutput();
}

#endif //CUTLASS_TESTS_H
