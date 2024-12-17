#include <iostream>
#include "main.h"
#include "isa.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"
#include "utils/program_utils.h"

__device__ bool getBitAt(uint8_t pixel_value, size_t bit_num) {
    if (bit_num >= 8) {
        printf("PD called more times than number of bits in image");
        return 0;
    } else {
        return pixel_value & (1 << bit_num); 
    }
}

__device__ void waitUntilAvailable(
    size_t* neighbour_program_counter,
    size_t neighbour_pc,
    size_t index,
    size_t image_size
) {
    while (index >= 0 && index < image_size && neighbour_program_counter[index] < neighbour_pc);
}

__device__ bool getInstructionInputValue(
    InputC inputc,
    bool *memory,
    uint8_t* image,
    size_t pd_bit,
    size_t image_x_dim,
    size_t image_size,
    size_t offset,
    size_t* neighbour_program_counter,
    bool* neighbour_shared_values,
    size_t neighbour_update_pc
) {
    bool input_value = false;
    switch (inputc.input.inputKind) {
        case InputKind::Address: input_value = memory[inputc.input.address]; break;
        case InputKind::ZeroValue: input_value = false; break;
        case InputKind::PD: input_value = getBitAt(image[offset], pd_bit); pd_bit++; break;
        case InputKind::Up:
            size_t up_index = offset - image_x_dim;
            waitUntilAvailable(neighbour_program_counter, neighbour_update_pc, up_index, image_size);
            input_value = (up_index >= 0) ? neighbour_shared_values[up_index] : 0;
            break;
        case InputKind::Down:
            size_t down_index = offset + image_x_dim;
            waitUntilAvailable(neighbour_program_counter, neighbour_update_pc, down_index, image_size);
            input_value = (down_index < image_size) ? neighbour_shared_values[down_index] : 0;
            break;
        case InputKind::Right:
            size_t right_index = offset + 1;
            waitUntilAvailable(neighbour_program_counter, neighbour_update_pc, right_index, image_size);
            input_value = (right_index < image_size) ? neighbour_shared_values[right_index] : 0;
            break;
        case InputKind::Left:
            size_t left_index = offset - 1;
            waitUntilAvailable(neighbour_program_counter, neighbour_update_pc, left_index, image_size);
            input_value = (left_index >= 0) ? neighbour_shared_values[left_index] : 0;
            break;
    }
    return (inputc.negated) ? ~input_value : input_value;
}

__global__ void processingElemKernel(
    Instruction* instructions,
    size_t num_instructions,
    uint8_t* image,
    bool* neighbour_shared_values,
    size_t* neighbour_program_counter,
    bool* external_values,
    size_t image_size,
    size_t image_x_dim,
    size_t image_y_dim,
    size_t num_outputs
) {
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t offset = x + y * blockDim.x * gridDim.x;

    if (offset < image_size) {
        const size_t MEMORY_SIZE_IN_BITS = 24;
        bool memory[MEMORY_SIZE_IN_BITS];
        for (size_t i = 0; i < MEMORY_SIZE_IN_BITS; i++) memory[i] = false;
        bool carry_register = false;
        size_t pd_bit = 0;
        size_t output_number = 0;

        // TODO DONT FORGET TO UPDATE
        size_t neighbour_update_pc = 0;

        for (size_t i = 0; i < num_instructions; i++) {
            const Instruction instruction = instructions[i];
            size_t pc = i + 1;
            bool carryval = false;
            switch (instruction.carry) {
                case Carry::CR: carryval = carry_register; break;
                case Carry::One: carryval = true; break;
                case Carry::Zero: carryval = false; break;
            }
            bool input_one = getInstructionInputValue(
                instruction.input1,
                memory,
                image,
                pd_bit,
                image_x_dim,
                image_size,
                offset,
                neighbour_program_counter,
                neighbour_shared_values,
                neighbour_update_pc
            );
            bool input_two = getInstructionInputValue(
                instruction.input2,
                memory,
                image,
                pd_bit,
                image_x_dim,
                image_size,
                offset,
                neighbour_program_counter,
                neighbour_shared_values,
                neighbour_update_pc
            );

            const bool sum = (input_one != input_two) != carryval;
            const bool carry = (carryval && (input_one != input_two)) || (input_one && input_two);

            // TODO assuming can only be two values
            bool resultvalue = (instruction.resultType.value == 's') ? sum : carry;

            // TODO Only update carry register when carry set to CR?
            carry_register = carry;

            switch (instruction.result.resultKind) {
                case ResultKind::Address:
                    memory[instruction.result.address] = resultvalue;
                    break;
                case ResultKind::Neighbour:
                    neighbour_update_pc = pc;
                    neighbour_shared_values[offset] = resultvalue;
                    neighbour_program_counter[offset] = pc;
                    break;
                case ResultKind::External:
                    external_values[num_outputs * offset + output_number] = resultvalue;
                    output_number++;
                    break;
            }
        }
    }
};

int main() {
    queryGPUProperties();

    std::string programFilename = "programs/prewitt.vis";
    std::string imageFilename = "";

    // Maximum of value below is 32
    size_t num_threads_per_block_per_dim = 16;

    size_t image_x_dim = 1;
    size_t image_y_dim = 1;
    size_t image_size = image_x_dim * image_y_dim;

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
    size_t image_mem_size = sizeof(uint8_t) * image_size;
    uint8_t* pixels = (uint8_t*) malloc(image_mem_size);
    
    // TODO For now, make all pixels one
    for (size_t i = 0; i < image_size; i++) {
        pixels[i] = 1;
    }
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

    cudaDeviceSynchronize();

    HANDLE_ERROR(cudaFree(dev_instructions));
    HANDLE_ERROR(cudaFree(dev_image));
    HANDLE_ERROR(cudaFree(dev_neighbour_shared_values));
    HANDLE_ERROR(cudaFree(dev_neighbour_program_counter));
    HANDLE_ERROR(cudaFree(dev_external_values));

    free(pixels);
    return EXIT_SUCCESS;
}