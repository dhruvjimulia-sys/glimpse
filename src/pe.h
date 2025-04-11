#pragma once

#include <cuda/atomic>
#include "isa.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"
#include "utils/program_utils.h"

// Note: MAX_NUM_INSTRUCTIONS set here (program.instructionCount * program.vliwWidth must be less than or equal to this value) 
#define MAX_NUM_INSTRUCTIONS 500
extern __constant__ char dev_instructions[sizeof(Instruction) * MAX_NUM_INSTRUCTIONS];

// Maximum of value below is 32
#define NUM_THREADS_PER_BLOCK_PER_DIM 16

__device__ __host__ bool getBitAt(uint8_t pixel_value, size_t bit_num);

__global__ void processingElemKernel(
    size_t num_instructions,
    uint8_t* image,
    bool* neighbour_shared_values,
    cuda::atomic<int, cuda::thread_scope_device>* neighbour_program_counter,
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
    bool is_pipelining
);