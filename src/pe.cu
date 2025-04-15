#include "pe.h"
 
__device__ __host__ bool getBitAt(uint8_t pixel_value, size_t bit_num) {
    if (bit_num >= 8) {
        printf("PD called more times than number of bits in image");
        return 0;
    } else {
        return (pixel_value & (1 << bit_num)) >> bit_num; 
    }
}

__device__ bool getNeighbourValue(
    bool* neighbour_shared_values,
    size_t neighbour_index,
    size_t num_shared_neighbours,
    size_t shared_neighbour_value
) {
    return neighbour_shared_values[neighbour_index * num_shared_neighbours + shared_neighbour_value - 1];
}

__device__ bool getInstructionInputValue(
    InputC inputc,
    bool* memory,
    uint8_t* image,
    size_t pd_bit,
    bool* pd_increment,
    int64_t x,
    int64_t y,
    size_t image_x_dim,
    size_t image_y_dim,
    size_t image_size,
    size_t offset,
    bool* neighbour_shared_values,
    size_t num_shared_neighbours,
    size_t shared_neighbour_value,
    bool use_shared_memory,
    bool* neighbour_shared_values_cache
) {
    bool input_value = false;
    switch (inputc.input.inputKind) {
        case InputKind::Address: input_value = memory[offset * MEMORY_SIZE_IN_BITS + inputc.input.address]; break;
        case InputKind::ZeroValue: input_value = false; break;
        case InputKind::PD:
            input_value = getBitAt(image[offset], pd_bit);
            *pd_increment = true;
            break;
        case InputKind::Up:
            if (y - 1 >= 0) {
                int64_t up_index = offset - image_x_dim;
                input_value = getNeighbourValue(
                    neighbour_shared_values,
                    up_index,
                    num_shared_neighbours,
                    shared_neighbour_value
                );
            } else {
                input_value = false;
            }
            break;
        case InputKind::Down:
            if (y + 1 < image_y_dim) {
                int64_t down_index = offset + image_x_dim;
                input_value = getNeighbourValue(
                    neighbour_shared_values,
                    down_index,
                    num_shared_neighbours,
                    shared_neighbour_value
                );
            } else {
                input_value = false;
            }
            break;
        case InputKind::Right:
            if (x + 1 < image_x_dim) {
                int64_t right_index = offset + 1;
                input_value = getNeighbourValue(
                    neighbour_shared_values,
                    right_index,
                    num_shared_neighbours,
                    shared_neighbour_value
                );
            } else {
                input_value = false;
            }
            break;
        case InputKind::Left:
            if (x - 1 >= 0) {
                int64_t left_index = offset - 1;
                input_value = getNeighbourValue(
                    neighbour_shared_values,
                    left_index,
                    num_shared_neighbours,
                    shared_neighbour_value
                );
            } else {
                input_value = false;
            }
            break;
        default:
            break;
    }
    return (inputc.negated) ? !input_value : input_value;
}

__global__ void processingElemKernel(
    size_t num_instructions,
    uint8_t* image,
    bool* neighbour_shared_values,
    bool* external_values,
    size_t image_size,
    size_t image_x_dim,
    size_t image_y_dim,
    size_t num_outputs,
    size_t num_shared_neighbours,
    size_t* debug_output,
    size_t num_debug_outputs,
    size_t vliw_width,
    bool use_shared_memory,
    bool is_pipelining,
    bool* local_memory_values,
    bool* carry_register_values,
    bool* result_values
) {
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t offset = x + y * blockDim.x * gridDim.x;
    // Note: PIPELINE_WIDTH
    const size_t PIPELINE_WIDTH = 3;
    
    if (offset < image_size) {
        cg::grid_group grid = cg::this_grid();
        
        // image_x, image_y in image space
        // x, y in thread/block space
        __shared__ bool neighbour_shared_values_cache[NUM_THREADS_PER_BLOCK_PER_DIM][NUM_THREADS_PER_BLOCK_PER_DIM];
        // if (use_shared_memory) {
        //     neighbour_shared_values_cache[threadIdx.y][threadIdx.x] = false;
        //     __syncthreads();
        // }
        size_t pd_bit = 0;
        bool pd_increment = false;
        size_t output_number = 0;

        // shared_neighbour_value is the index of the shared neighbour value
        bool shared_neighbour_value_increment = false;
        size_t shared_neighbour_value = 0;

        bool output_number_increment = false;

        bool contains_neighbour_sharing = false;

        for (size_t i = 0; (i < num_instructions && !is_pipelining) || (i < num_instructions + PIPELINE_WIDTH - 1 && is_pipelining); i++) {
            size_t offset = x + y * blockDim.x * gridDim.x;
            size_t image_x = offset % image_x_dim;
            size_t image_y = offset / image_x_dim;
            while (offset < image_size) {
                if (i < num_instructions) {
                    for (size_t j = 0; j < vliw_width; j++) { 
                        const Instruction instruction = ((Instruction *) dev_instructions)[i * vliw_width + j];
                        if (instruction.isNop) {
                            continue;
                        }
                        bool carryval = false;
                        switch (instruction.carry) {
                            case Carry::CR: carryval = carry_register_values[offset * vliw_width + j]; break;
                            case Carry::One: carryval = true; break;
                            case Carry::Zero: carryval = false; break;
                        }
                        bool input_one = getInstructionInputValue(
                            instruction.input1,
                            local_memory_values,
                            image,
                            pd_bit,
                            &pd_increment,
                            image_x,
                            image_y,
                            image_x_dim,
                            image_y_dim,
                            image_size,
                            offset,
                            neighbour_shared_values,
                            num_shared_neighbours,
                            shared_neighbour_value,
                            use_shared_memory,
                            (bool *) neighbour_shared_values_cache
                        );
                        bool input_two = getInstructionInputValue(
                            instruction.input2,
                            local_memory_values,
                            image,
                            pd_bit,
                            &pd_increment,
                            image_x,
                            image_y,
                            image_x_dim,
                            image_y_dim,
                            image_size,
                            offset,
                            neighbour_shared_values,
                            num_shared_neighbours,
                            shared_neighbour_value,
                            use_shared_memory,
                            (bool *) neighbour_shared_values_cache
                        );

                        // printf("offset: %lu, instruction: %lu, input_one: %d, carryval: %d, input_two: %d\n", offset, i, input_one, carryval, input_two);
                        
                        // debug_output value = 0 if nop
                        // debug_output[((offset * num_instructions + i) * vliw_width + j) * num_debug_outputs] = input_one;
                        // debug_output[((offset * num_instructions + i) * vliw_width + j) * num_debug_outputs + 1] = input_two;
                        // debug_output[((offset * num_instructions + i) * vliw_width + j) * num_debug_outputs + 2] = carryval;

                        const bool sum = (input_one != input_two) != carryval;
                        const bool carry = (carryval && (input_one != input_two)) || (input_one && input_two);

                        // Assuming can only be two values
                        result_values[(offset * vliw_width + j) * PIPELINE_WIDTH + (i % PIPELINE_WIDTH)] = (instruction.resultType.value == 's') ? sum : carry;

                        // Interesting choice...
                        if (instruction.carry == Carry::CR) {
                            carry_register_values[offset * vliw_width + j] = carry;
                        }
                    }
                }

                if (!is_pipelining || (is_pipelining && i >= PIPELINE_WIDTH - 1)) {
                    for (size_t j = 0; j < vliw_width; j++) {
                        const Instruction instruction = 
                        !is_pipelining ?
                        ((Instruction *) dev_instructions)[i * vliw_width + j] :
                        ((Instruction *) dev_instructions)[(i - PIPELINE_WIDTH + 1) * vliw_width + j];
                        if (instruction.isNop) {
                            continue;
                        }
                        size_t resultvalue = 
                        !is_pipelining ?
                        result_values[(offset * vliw_width + j) * PIPELINE_WIDTH + (i % PIPELINE_WIDTH)] :
                        result_values[(offset * vliw_width + j) * PIPELINE_WIDTH + ((i - PIPELINE_WIDTH + 1) % PIPELINE_WIDTH)];
                        switch (instruction.result.resultKind) {
                            case ResultKind::Address:
                                local_memory_values[offset * MEMORY_SIZE_IN_BITS + instruction.result.address] = resultvalue;
                                break;
                            case ResultKind::Neighbour:
                                neighbour_shared_values[offset * num_shared_neighbours + shared_neighbour_value] = resultvalue;
                                shared_neighbour_value_increment = true;
                                contains_neighbour_sharing = true;
                                break;
                            case ResultKind::External:
                                external_values[num_outputs * offset + output_number] = resultvalue;
                                output_number_increment = true;
                                break;
                        }
                    }
                }
                offset += blockDim.x * gridDim.x * blockDim.y * gridDim.y;
                image_x = offset % image_x_dim;
                image_y = offset / image_x_dim;
            }

            if (pd_increment) {
                pd_bit++;
            }
            pd_increment = false;
            if (shared_neighbour_value_increment) {
                shared_neighbour_value++;
            }
            shared_neighbour_value_increment = false;
            if (contains_neighbour_sharing) {
                grid.sync();
            }
            contains_neighbour_sharing = false;
            if (output_number_increment) {
                output_number++;
            }
            output_number_increment = false;
        }
    }
};
