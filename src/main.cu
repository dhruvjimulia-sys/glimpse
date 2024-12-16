#include <iostream>
#include "main.h"
#include "../include/isa.h"
#include "../include/utils.h"
#include "../include/file_utils.h"

__global__ void processingElemKernel(void) {
    
};

int main() {
    std::string programFilename = "programs/prewitt.vis";
    std::string programText;
    readFile(programFilename, programText);

    Parser parser(programText);
    std::shared_ptr<Program> program = parser.parse();
    program->print();

    // read instructions from file, parse and memcpy to cuda (constant) memory


    // read grayscale pixels from image and memcpy to cuda (constant) memory

    // cudamalloc memory for each processing elem (including neighbour latch?)


    // cudamalloc memory for program counter when neighbour written
    
    processingElemKernel<<<1, 1>>>();

    // cuda free

    return 0;
}