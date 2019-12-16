//
// Created by Egor Orachyov on 2019-12-10.
//

#ifndef EXPERIMENTAL_GRAPH_H
#define EXPERIMENTAL_GRAPH_H

#include <vector>

/// Stores graph data
class GraphData {
public:

    const std::vector<int> &getMatrix() const {
        return matrix;
    }

    int getVerticesCount() const {
        return verticesCount;
    }

private:

    friend class AutomataLoader;
    int verticesCount = 0;
    std::vector<int> matrix;

};


#endif //EXPERIMENTAL_GRAPH_H
