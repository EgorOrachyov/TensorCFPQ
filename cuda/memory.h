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

#ifndef COMPUTATION_MEMORY_H
#define COMPUTATION_MEMORY_H

#include "misc.h"

// Memory utils for work with cuda device

void gpu_allocate(void** p, size_t size) {
    checkCudaError(cudaMalloc(p, size), "gpu_allocate");
}

void gpu_free(void* p) {
    checkCudaError(cudaFree(p), "gpu_free");
}

void gpu_copyDeviceToHost(void* dest, const void* src, size_t size) {
    checkCudaError(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost), "gpu_copyDeviceToHost");
}

void gpu_copyDeviceToDevice(void* dest, const void* src, size_t size) {
    checkCudaError(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice), "gpu_copyDeviceToDevice");
}

void gpu_copyHostToDevice(void* dest, const void* src, size_t size) {
    checkCudaError(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice), "gpu_copyHostToDevice");
}

void gpu_memorySet(void* p, int value, size_t size) {
    checkCudaError(cudaMemset(p, value, size), "gpu_memorySet");
}

#endif // COMPUTATION_MEMORY_H
