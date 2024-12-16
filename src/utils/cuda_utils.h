#pragma once

#define HANDLE_ERROR(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(error); \
    } \
}

void queryGPUProperties();