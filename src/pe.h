#pragma once

#include <cooperative_groups.h>

#include <cuda/atomic>
namespace cg = cooperative_groups;
#include "isa.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"
#include "utils/program_utils.h"

// Note: MAX_NUM_INSTRUCTIONS set here (program.instructionCount *
// program.vliwWidth must be less than or equal to this value)
#define MAX_NUM_INSTRUCTIONS 500
// Note: MEMORY_SIZE_IN_BITS
#define MEMORY_SIZE_IN_BITS 24
extern __constant__ char
    dev_instructions[sizeof(Instruction) * MAX_NUM_INSTRUCTIONS];

// Maximum of value below is 32

// TODO Change
#define NUM_THREADS_PER_BLOCK_PER_DIM 16

__device__ __host__ bool getBitAt(uint8_t pixel_value, size_t bit_num);

__global__ void processingElemKernel(
    size_t num_instructions, uint8_t* image, bool* neighbour_shared_values,
    bool* external_values, size_t image_size, size_t image_x_dim,
    size_t image_y_dim, size_t num_outputs, size_t num_shared_neighbours,
    size_t* debug_output, size_t num_debug_outputs, size_t vliw_width,
    bool use_shared_memory, bool is_pipelining, bool* local_memory_values,
    bool* carry_register_values, bool* result_values);