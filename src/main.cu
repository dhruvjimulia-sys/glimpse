#include <iostream>
#include "main.h"
#include "isa.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"

__global__ void processingElemKernel(Instruction* instructions, size_t num_instructions) {
    instructions[0].print();
    // for (int i = 0; i < num_instructions; i++) {
    //     instructions[i].print();
    // }
};

int main() {
    std::string programFilename = "programs/prewitt.vis";
    std::string programText;
    readFile(programFilename, programText);

    Parser parser(programText);
    Program program = parser.parse();
    program.print();

    // read instructions from file, parse and memcpy to cuda (constant) memory
    Instruction* dev_instructions;
    size_t instructions_size = sizeof(Instruction) * program.instructionCount;
    HANDLE_ERROR(cudaMalloc((void **) &dev_instructions, instructions_size));
    HANDLE_ERROR(cudaMemcpy(dev_instructions, program.instructions, instructions_size, cudaMemcpyHostToDevice));

    // read grayscale pixels from image and memcpy to cuda (constant) memory

    // cudamalloc memory for each processing elem (including neighbour latch?)


    // cudamalloc memory for program counter when neighbour written
    
    processingElemKernel<<<1, 1>>>(dev_instructions, program.instructionCount);

    cudaDeviceSynchronize();
    // cuda free


    return 0;
}