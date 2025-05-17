#pragma once

#include <cooperative_groups.h>

#include <cuda/atomic>
namespace cg = cooperative_groups;
#include "isa.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"
#include "utils/program_utils.h"

// Maximum of value below is 32
// Finetuned to max possible
#define NUM_THREADS_PER_BLOCK_PER_DIM 20

__device__ __host__ bool getBitAt(uint8_t pixel_value, size_t bit_num);

__global__ void processingElemKernel(
    Instruction* instructions, size_t num_instructions, uint8_t* image,
    bool* neighbour_shared_values, bool* external_values, size_t image_size,
    size_t image_x_dim, size_t image_y_dim, size_t num_outputs,
    size_t num_shared_neighbours, size_t* debug_output,
    size_t num_debug_outputs, size_t vliw_width, bool is_pipelining,
    bool* local_memory_values, bool* carry_register_values, bool* result_values,
    size_t num_iterations);