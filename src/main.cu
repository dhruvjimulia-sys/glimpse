#include <iostream>
#include "main.h"
#include "isa.h"
#include "pe.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"
#include "utils/program_utils.h"

int main() {
    queryGPUProperties();

    std::string programFilename = "programs/edge_detection_one_bit.vis";
    std::string imageFilename = "";

    // Maximum of value below is 32
    size_t num_threads_per_block_per_dim = 16;

    // read instructions from file, parse and memcpy to cuda memory
    // TODO make this CUDA memory constant as optimization
    std::string programText;
    readFile(programFilename, programText);

    Parser parser(programText);
    Program program = parser.parse();
    program.print();

    size_t program_num_outputs = numOutputs(program);

    Instruction* dev_instructions;
    size_t instructions_mem_size = sizeof(Instruction) * program.instructionCount;
    HANDLE_ERROR(cudaMalloc((void **) &dev_instructions, instructions_mem_size));
    HANDLE_ERROR(cudaMemcpy(dev_instructions, program.instructions, instructions_mem_size, cudaMemcpyHostToDevice));

    // read grayscale pixels from image and memcpy to cuda memory
    // TODO make this CUDA memory constant as optimization
    size_t image_x_dim = 20;
    size_t image_y_dim = 20;
    size_t image_size = image_x_dim * image_y_dim;
    
    size_t image_mem_size = sizeof(uint8_t) * image_size;
    // uint8_t* pixels = (uint8_t*) malloc(image_mem_size);
    
    // // TODO For now, make all pixels one
    // for (size_t i = 0; i < image_size; i++) {
    //     pixels[i] = 1;
    // }

    uint8_t pixels[20][20] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
    };

    // uint8_t pixels[20][20] = {
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    // };

    uint8_t* dev_image;

    HANDLE_ERROR(cudaMalloc((void **) &dev_image, image_mem_size));
    HANDLE_ERROR(cudaMemcpy(dev_image, pixels, image_mem_size, cudaMemcpyHostToDevice));

    // neighbour
    bool* dev_neighbour_shared_values;
    size_t neighbour_shared_mem_size = sizeof(bool) * image_size;
    HANDLE_ERROR(cudaMalloc((void **) &dev_neighbour_shared_values, neighbour_shared_mem_size));
    HANDLE_ERROR(cudaMemset(dev_neighbour_shared_values, 0, neighbour_shared_mem_size));

    // program counter when neighbour written
    size_t* dev_neighbour_program_counter;
    size_t neighbour_program_counter_mem_size = sizeof(size_t) * image_size;
    HANDLE_ERROR(cudaMalloc((void **) &dev_neighbour_program_counter, neighbour_program_counter_mem_size));
    HANDLE_ERROR(cudaMemset(dev_neighbour_program_counter, 0, neighbour_program_counter_mem_size));

    // external values
    bool* dev_external_values;
    size_t external_values_mem_size = sizeof(bool) * image_size * program_num_outputs;
    HANDLE_ERROR(cudaMalloc((void **) &dev_external_values, external_values_mem_size));
    HANDLE_ERROR(cudaMemset(dev_external_values, 0, external_values_mem_size));

    // TODO generalize to arbitrary length messages
    dim3 blocks(
        (image_x_dim + num_threads_per_block_per_dim - 1) / num_threads_per_block_per_dim,
        (image_y_dim + num_threads_per_block_per_dim - 1) / num_threads_per_block_per_dim
    );
    dim3 threads(num_threads_per_block_per_dim, num_threads_per_block_per_dim);
    processingElemKernel<<<blocks, threads>>>(
        dev_instructions,
        program.instructionCount,
        dev_image,
        dev_neighbour_shared_values,
        dev_neighbour_program_counter,
        dev_external_values,
        image_size,
        image_x_dim,
        image_y_dim,
        program_num_outputs
    );

    HANDLE_ERROR(cudaDeviceSynchronize());

    bool* external_values = (bool *) malloc(external_values_mem_size);
    HANDLE_ERROR(cudaMemcpy(external_values, dev_external_values, external_values_mem_size, cudaMemcpyDeviceToHost));

    for (size_t y = 0; y < image_y_dim; y++) {
        for (size_t x = 0; x < image_x_dim; x++) {
            size_t offset = x + y * image_x_dim;
            for (int64_t i = program_num_outputs - 1; i >= 0; i--) {
                printf("%d", external_values[program_num_outputs * offset + i]);
            }
            printf(" ");
        }
        printf("\n");
    }

    HANDLE_ERROR(cudaFree(dev_instructions));
    HANDLE_ERROR(cudaFree(dev_image));
    HANDLE_ERROR(cudaFree(dev_neighbour_shared_values));
    HANDLE_ERROR(cudaFree(dev_neighbour_program_counter));
    HANDLE_ERROR(cudaFree(dev_external_values));

    // free(pixels);
    return EXIT_SUCCESS;
}