//
// Created by Egor Orachyov on 2019-12-10.
//

#ifndef EXPERIMENTAL_TIMER_H
#define EXPERIMENTAL_TIMER_H

#include <chrono>

class Timer {
public:

    void begin() {
        start = std::chrono::high_resolution_clock::now();
    }

    void end() {
        finish = std::chrono::high_resolution_clock::now();
    }

    /// @return Elapsed time between begin() and end() calls
    double elapsed() {
        return std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / 1000.0f / 1000.0f;
    }

private:

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;

};


#endif //EXPERIMENTAL_TIMER_H
