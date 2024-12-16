#include <iostream>
#include "isa.h"

#define HANDLE_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "Error: " << cudaGetErrorString(err) << "\n"; \
        return; \
    }

__global__ void processingElemKernel(void) {
    
};

void queryGPUProperties() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    HANDLE_ERROR(err);

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return;
    }

    std::cout << "Number of CUDA-capable devices: " << deviceCount << std::endl;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, device));

        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA Cores/MP: " << (deviceProp.major == 8 ? 128 : 64) << " (Architecture-dependent)" << std::endl;
        std::cout << "  Total CUDA Cores: " << deviceProp.multiProcessorCount * (deviceProp.major == 8 ? 128 : 64) << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread dimensions: ["
                  << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", "
                  << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max grid size: ["
                  << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", "
                  << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak memory bandwidth: "
                  << 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
    }
}

int main() {
    queryGPUProperties();

    // read instructions from file and memcopy to cuda memory

    // read grayscale pixels from image and memcpy to cuda (constant) memory

    // cudamalloc memory for each processing elem

    // cudamalloc memory for program counter when neighbour written
    
    processingElemKernel<<<1, 1>>>();

    return 0;
}