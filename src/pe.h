#pragma once

#include <cuda/atomic>
#include "isa.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"
#include "utils/program_utils.h"

__global__ void processingElemKernel(
    Instruction* instructions,
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
    size_t num_debug_outputs
);