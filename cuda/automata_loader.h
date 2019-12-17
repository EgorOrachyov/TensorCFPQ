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

#ifndef EXPERIMENTAL_AUTOMATA_H
#define EXPERIMENTAL_AUTOMATA_H

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "graph_data.h"

/// Load and process recursive automata.
/// Loads graph data, verified accordingly to automata content
class AutomataLoader {
public:

    AutomataLoader(const std::string& filename) {
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << "\n";
            exit(10);
        }

        std::string label;
        int edgesCount = 0;
        int pathsCount = 0;

        file >> edgesCount;
        edges.reserve(edgesCount);
        for (int t = 0; t < edgesCount; t++) {
            int i = 0, j = 0, l = 0;

            file >> i >> label >> j;

            auto found = indices.find(label);
            if (found != indices.end()) {
                l = found->second;
            } else {
                l = 1 << indicesCount;
                indicesCount += 1;
                indices.emplace(label, l);
                labels.emplace(l, label);
            }

            edges.push_back({ i, j , l });

            statesCount = (i > statesCount ? i : statesCount);
            statesCount = (j > statesCount ? j : statesCount);
        }

        statesCount += 1;

        file >> pathsCount;
        paths.reserve(pathsCount);
        for (int t = 0; t < pathsCount; t++) {
            int i = 0, j = 0, l = 0;

            file >> i >> label >> j;

            auto found = indices.find(label);
            if (found != indices.end()) {
                l = found->second;
            } else {
//                std::cerr << "Automata has incomplete type [name: " << filename << "]\n";
//                continue;
                l = 1 << indicesCount;
                indicesCount += 1;
                indices.emplace(label, l);
                labels.emplace(l, label);
            }

            paths.push_back({ i, j , l });
        }

        file >> startSymbolsCount;
        startSymbols.reserve(startSymbolsCount);
        for (int t = 0; t < startSymbolsCount; t++) {
            int l = 0;
            file >> label;

            auto found = indices.find(label);
            if (found != indices.end()) {
                l = found->second;
            } else {
//                std::cerr << "Automata has incomplete type [name: " << filename << "]\n";
//                continue;
                l = 1 << indicesCount;
                indicesCount += 1;
                indices.emplace(label, l);
                labels.emplace(l, label);
            }

            startSymbols.push_back(l);
        }

        file.close();

        {
            matrix.reserve(statesCount * statesCount);
            states.reserve(statesCount * statesCount);
            for (int i = 0; i < statesCount * statesCount; i++) {
                matrix.push_back(0);
                states.push_back(0);
            }

            for (auto& e: edges) {
                matrix[e.i * statesCount + e.j] |= e.label;
            }

            for (auto& p: paths) {
                states[p.i * statesCount + p.j] |= p.label;
            }
        }

#if 0
        {
            std::cout << statesCount << "\n";

            for (int i = 0; i < statesCount; i++) {
                for (int j = 0; j < statesCount; j++) {
                    std::cout << matrix[i * statesCount + j] << " ";
                }
                std::cout << "\n";
            }

            for (int i = 0; i < statesCount; i++) {
                for (int j = 0; j < statesCount; j++) {
                    std::cout << states[i * statesCount + j] << " ";
                }
                std::cout << "\n";
            }

            for (auto& e: edges) {
                std::cout << e.i << " " << e.label << " " << e.j << "\n";
            }

            for (auto& p: paths) {
                std::cout << p.i << " " << p.label << " " << p.j << "\n";
            }

            for (auto& i: startSymbols) {
                std::cout << i << "\n";
            }
        }
#endif
    }

    const std::vector<int> &getMatrix() const {
        return matrix;
    }

    const std::vector<int> getStates() const {
        return states;
    }

    const std::vector<int> getStartSymbols() const {
        return startSymbols;
    }

    int getStatesCount() const {
        return statesCount;
    }

    int getStartSymbolsCount() const {
        return startSymbolsCount;
    }

    /// Loads specified graph from file
    /// @param filename File to load graph data
    /// @param graph Output stored graph data
    void loadGraph(const std::string& filename, GraphData& graph) {
        std::ifstream file(filename);
        std::vector<edge> graphEdges;
        std::string label;

        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << "\n";
            exit(10);
        }

        int verticesCount = 0;

        while (!file.eof()) {
            int i = 0, j = 0, l = 0;

            file >> i >> label >> j;

            auto found = indices.find(label);
            if (found != indices.end()) {
                l = found->second;
            } else {
                // std::cout << "Unexpected symbol: " << label << "\n";
                // add 0 labels where we have some unused symbols
                l = 0;
            }

            graphEdges.push_back({ i, j , l });

            verticesCount = (i > verticesCount ? i : verticesCount);
            verticesCount = (j > verticesCount ? j : verticesCount);
        }

        verticesCount += 1;

        file.close();

        auto &graphMatrix = graph.matrix;
        graph.verticesCount = verticesCount;

        {
            graphMatrix.reserve(verticesCount * verticesCount);
            for (int i = 0; i < verticesCount * verticesCount; i++) {
                graphMatrix.push_back(0);
            }

            for (auto& e: graphEdges) {
                graphMatrix[e.i * verticesCount + e.j] |= e.label;
            }
        }

        {
            for (int i = 0; i < statesCount; i++) {
                int s = states.data()[i * statesCount + i];

                if (s != 0) {
                    for (int j = 0; j < verticesCount; j++) {
                        graphMatrix[j * verticesCount + j] |= s;
                    }
                }
            }
        }

#if 0
        {
            std::cout << verticesCount << "\n";

            for (auto& e: graphEdges) {
                std::cout << e.i << " " << e.label << " " << e.j << "\n";
            }

            for (int i = 0; i < verticesCount; i++) {
                for (int j = 0; j < verticesCount; j++) {
                    std::cout << graphMatrix[i * verticesCount + j] << " ";
                }
                std::cout << "\n";
            }
        }
#endif
    }

    void generateGraphVis(std::string filename, const int* graph, int N) {
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << "\n";
            exit(10);
        }

        file << "digraph generated {";

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int e = graph[i * N + j];

                if (e != 0x0) {
                    bool write = false;
                    for (int t = 1; t != 0; t = t << 1) {
                        if ((e & t) != 0) {
                            auto label = labels.find(t);
                            if (label != labels.end()) {
                                if (write) {
                                    file << "," << label->second;
                                } else {
                                    file << i << " -> " << j << " [ label = \"";
                                    file << label->second;
                                    write = true;
                                }
                            }
                        }
                    }

                    if (write) {
                        file << "\" ]; ";
                    }
                }
            }
        }

        file << "}";
        file.close();
    }

private:

    struct edge {
        int i;
        int j;
        int label;
    };

    int indicesCount = 0;
    int statesCount = 0;
    int startSymbolsCount = 0;



    /// Automata transition matrix
    std::vector<int> matrix;

    /// States matrix, describes path of derivation of non-terminals
    std::vector<int> states;

    /// Start non-terminals
    std::vector<int> startSymbols;

    /// Loaded automata edges from file
    std::vector<edge> edges;

    /// Loaded paths of non-termainal derivation from file
    std::vector<edge> paths;

    /// Indices of mapping string represented terminals/non-terminals to int bit indices
    std::map<std::string, int> indices;

    /// Map indices into labels
    std::map<int, std::string> labels;

};

#endif //EXPERIMENTAL_AUTOMATA_H
