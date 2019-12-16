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
