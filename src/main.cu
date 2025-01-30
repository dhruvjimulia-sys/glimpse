#include <iostream>
#include "main.h"
#include "isa.h"
#include "pe.h"
#include "utils/cuda_utils.h"
#include "utils/file_utils.h"
#include "utils/program_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"


uint8_t* transform_image(const char* filename, int new_dimension, int new_bits) {
    int width, height, channels;
    uint8_t* img_data = stbi_load(filename, &width, &height, &channels, 0);
    if (!img_data) {
        return nullptr;
    }

    if (new_bits < 1 || new_bits > 8) {
        stbi_image_free(img_data);
        return nullptr;
    }

    // TODO Make resize automatic
    uint8_t* resized_data = img_data;
    
    // (uint8_t*)malloc(new_dimension * new_dimension * channels);
    // if (!resized_data) {
    //     stbi_image_free(img_data);
    //     return nullptr;
    // }

    // // Resize the image
    // stbir_resize_uint8(img_data, width, height, 0,
    //                    resized_data, new_dimension, new_dimension, 0,
    //                    channels);
    // stbi_image_free(img_data); // Free original image data

    // Convert to grayscale (1 channel)
    uint8_t* gray_data = (uint8_t*)malloc(new_dimension * new_dimension);
    if (!gray_data) {
        free(resized_data);
        return nullptr;
    }

    for (int i = 0; i < new_dimension * new_dimension; ++i) {
        int src_idx = i * channels;
        if (channels >= 3) {
            // Use luminance formula: 0.299*R + 0.587*G + 0.114*B (integer approximation)
            uint8_t r = resized_data[src_idx];
            uint8_t g = resized_data[src_idx + 1];
            uint8_t b = resized_data[src_idx + 2];
            gray_data[i] = static_cast<uint8_t>((r * 299 + g * 587 + b * 114 + 500) / 1000);
        } else {
            gray_data[i] = resized_data[src_idx];
        }
    }
    free(resized_data); 

    // Quantized to required bit depth
    const int max_level = (1 << new_bits) - 1;
    if (max_level > 0) {
        for (int i = 0; i < new_dimension * new_dimension; ++i) {
            gray_data[i] = (gray_data[i] >> (8 - new_bits)) & max_level;
        }
    }

    return gray_data;
}

bool *processImage(Program program, uint8_t* pixels, size_t image_x_dim, size_t image_y_dim) {
    size_t program_num_outputs = numOutputs(program);

    // Maximum of value below is 32
    size_t num_threads_per_block_per_dim = 16;
    
    // TODO make this CUDA memory constant as optimization
    Instruction* dev_instructions;
    size_t instructions_mem_size = sizeof(Instruction) * program.instructionCount;
    HANDLE_ERROR(cudaMalloc((void **) &dev_instructions, instructions_mem_size));
    HANDLE_ERROR(cudaMemcpy(dev_instructions, program.instructions, instructions_mem_size, cudaMemcpyHostToDevice));

    // read grayscale pixels from image and memcpy to cuda memory
    // TODO make this CUDA memory constant as optimization

    size_t image_size = image_x_dim * image_y_dim;
    
    size_t image_mem_size = sizeof(uint8_t) * image_size;

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

    HANDLE_ERROR(cudaPeekAtLastError());

    HANDLE_ERROR(cudaDeviceSynchronize());

    bool* external_values = (bool *) malloc(external_values_mem_size);
    HANDLE_ERROR(cudaMemcpy(external_values, dev_external_values, external_values_mem_size, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_instructions));
    HANDLE_ERROR(cudaFree(dev_image));
    HANDLE_ERROR(cudaFree(dev_neighbour_shared_values));
    HANDLE_ERROR(cudaFree(dev_neighbour_program_counter));
    HANDLE_ERROR(cudaFree(dev_external_values));

    return external_values;
}


int main() {
    queryGPUProperties();

    std::string programFilename = "programs/edge_detection_one_bit.vis";
    const char *imageFilename = "images/windmill_resized.jpg";

    size_t dimension = 128;
    size_t num_bits = 1;

    uint8_t* image = transform_image(imageFilename, dimension, num_bits);
    
    // print image
    // for (size_t i = 0; i < dimension; i++) {
    //     for (size_t j = 0; j < dimension; j++) {
    //         printf("%d ", image[i * dimension + j]);
    //     }
    //     printf("\n");
    // }

    std::string programText;
    readFile(programFilename, programText);

    Parser parser(programText);
    Program program = parser.parse();
    program.print();

    size_t program_num_outputs = numOutputs(program);

    std::vector<std::vector<std::vector<bool>>> expected_image(dimension, std::vector<std::vector<bool>>(dimension, std::vector<bool>(program_num_outputs, 0)));
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            size_t val = image[i * dimension + j];
            expected_image[i][j][0] =
            (((i - 1 < 0) ? 0 : image[(i - 1) * dimension + j]) != val)
            || (((i + 1 >= dimension) ? 0 : image[(i + 1) * dimension + j]) != val)
            || (((j - 1 < 0) ? 0 : image[i * dimension + (j - 1)]) != val)
            || (((j + 1 >= dimension) ? 0 : image[i * dimension + j + 1]) != val); 
        }
    }

    bool* processed_image = processImage(program, image, dimension, dimension);

    bool test_passed = true;
    for (size_t y = 0; y < dimension; y++) {
        for (size_t x = 0; x < dimension; x++) {
            size_t offset = x + y * dimension;
            for (int64_t i = program_num_outputs - 1; i >= 0; i--) {
                bool actual_value = processed_image[program_num_outputs * offset + i];
                if (actual_value != expected_image[y][x][i]) {
                    test_passed = false;
                }
            }
        }
    }

    if (test_passed) {
        std::cout << "Test passed" << std::endl;
    } else {
        std::cout << "Test failed" << std::endl;
    }

    free(image);
    free(program.instructions);
    return EXIT_SUCCESS;
}