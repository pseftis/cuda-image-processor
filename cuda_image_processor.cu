// CUDA Image Processor - Gaussian Blur Filter
// Applies a Gaussian blur filter to images using CUDA parallel processing

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Structure to hold image data
struct Image {
    int width;
    int height;
    int channels;
    unsigned char* data;
};

// Read PPM image file (P6 format)
Image ReadPPM(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    std::string format;
    file >> format;
    if (format != "P6") {
        fprintf(stderr, "Error: Unsupported format. Expected P6 PPM\n");
        exit(1);
    }

    Image img;
    file >> img.width >> img.height;
    int max_val;
    file >> max_val;
    file.get(); // consume newline

    img.channels = 3; // RGB
    int size = img.width * img.height * img.channels;
    img.data = new unsigned char[size];
    file.read(reinterpret_cast<char*>(img.data), size);
    file.close();

    return img;
}

// Write PPM image file (P6 format)
void WritePPM(const char* filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        exit(1);
    }

    file << "P6\n";
    file << img.width << " " << img.height << "\n";
    file << "255\n";
    file.write(reinterpret_cast<const char*>(img.data),
                img.width * img.height * img.channels);
    file.close();
}

// CUDA kernel for Gaussian blur
__global__ void GaussianBlurKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    float* gaussian_kernel,
    int kernel_size) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int radius = kernel_size / 2;
    float sum[3] = {0.0f, 0.0f, 0.0f};
    float weight_sum = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
            int px = x + kx;
            int py = y + ky;

            // Handle boundary conditions
            if (px < 0) px = 0;
            if (px >= width) px = width - 1;
            if (py < 0) py = 0;
            if (py >= height) py = height - 1;

            int kernel_idx = (ky + radius) * kernel_size + (kx + radius);
            float weight = gaussian_kernel[kernel_idx];

            int pixel_idx = (py * width + px) * channels;
            for (int c = 0; c < channels; c++) {
                sum[c] += input[pixel_idx + c] * weight;
            }
            weight_sum += weight;
        }
    }

    int out_idx = (y * width + x) * channels;
    for (int c = 0; c < channels; c++) {
        output[out_idx + c] = (unsigned char)(sum[c] / weight_sum);
    }
}

// Generate Gaussian kernel
void GenerateGaussianKernel(float* kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    float two_sigma_sq = 2.0f * sigma * sigma;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float dist_sq = x * x + y * y;
            float value = expf(-dist_sq / two_sigma_sq);
            int idx = (y + radius) * size + (x + radius);
            kernel[idx] = value;
            sum += value;
        }
    }

    // Normalize
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// Process image using CUDA
void ProcessImageCUDA(const Image& input, Image& output,
                           int kernel_size, float sigma) {
    // Allocate device memory
    size_t image_size = input.width * input.height * input.channels;
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_kernel;

    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data, image_size,
                          cudaMemcpyHostToDevice));

    // Generate and copy kernel
    float* h_kernel = new float[kernel_size * kernel_size];
    GenerateGaussianKernel(h_kernel, kernel_size, sigma);
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel,
                          kernel_size * kernel_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Configure kernel launch
    dim3 block_size(16, 16);
    dim3 grid_size((input.width + block_size.x - 1) / block_size.x,
                   (input.height + block_size.y - 1) / block_size.y);

    // Launch kernel
    GaussianBlurKernel<<<grid_size, block_size>>>(
        d_input, d_output, input.width, input.height,
        input.channels, d_kernel, kernel_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output.data, d_output, image_size,
                          cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
    delete[] h_kernel;
}

// Print usage information
void PrintUsage(const char* program_name) {
    printf("Usage: %s <input_file> <output_file> [options]\n", program_name);
    printf("Options:\n");
    printf("  --kernel-size N    Gaussian kernel size (default: 15, must be odd)\n");
    printf("  --sigma F          Gaussian sigma value (default: 3.0)\n");
    printf("  --help             Show this help message\n");
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        PrintUsage(argv[0]);
        return 1;
    }

    // Parse command line arguments
    const char* input_file = nullptr;
    const char* output_file = nullptr;
    int kernel_size = 15;
    float sigma = 3.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--kernel-size") == 0 && i + 1 < argc) {
            kernel_size = atoi(argv[++i]);
            if (kernel_size % 2 == 0) {
                fprintf(stderr, "Error: Kernel size must be odd\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            sigma = atof(argv[++i]);
        } else if (!input_file) {
            input_file = argv[i];
        } else if (!output_file) {
            output_file = argv[i];
        }
    }

    if (!input_file || !output_file) {
        fprintf(stderr, "Error: Input and output files must be specified\n");
        PrintUsage(argv[0]);
        return 1;
    }

    // Print CUDA device information
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\n");

    // Read input image
    printf("Reading image: %s\n", input_file);
    Image input = ReadPPM(input_file);
    printf("Image dimensions: %d x %d\n", input.width, input.height);

    // Allocate output image
    Image output;
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data = new unsigned char[input.width * input.height * input.channels];

    // Process image
    printf("Processing image with kernel size %d, sigma %.2f...\n",
           kernel_size, sigma);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    ProcessImageCUDA(input, output, kernel_size, sigma);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Processing completed in %.2f ms\n", milliseconds);
    printf("Throughput: %.2f MPixels/s\n",
           (input.width * input.height) / (milliseconds / 1000.0f) / 1e6);

    // Write output image
    printf("Writing output: %s\n", output_file);
    WritePPM(output_file, output);

    // Cleanup
    delete[] input.data;
    delete[] output.data;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("Done!\n");
    return 0;
}

