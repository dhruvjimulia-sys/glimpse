#include <iostream>
#include "../include/isa.h"
#include "../include/utils.h"

__global__ void processingElemKernel(void) {
    
};

int main() {
    queryGPUProperties();

    // read instructions from file, parse and memcopy to cuda memory

    // read grayscale pixels from image and memcpy to cuda (constant) memory

    // cudamalloc memory for each processing elem

    // cudamalloc memory for program counter when neighbour written
    
    processingElemKernel<<<1, 1>>>();

    // cuda free

    return 0;
}