# CUDA Image Processor - Gaussian Blur Filter

## Project Description

This project implements a high-performance Gaussian blur filter for images using NVIDIA CUDA parallel computing. The application demonstrates scalable GPU acceleration for image processing tasks, capable of processing large images efficiently by leveraging thousands of parallel threads on the GPU.

The Gaussian blur filter is a fundamental image processing operation that smooths images by averaging pixel values with their neighbors, weighted by a Gaussian distribution. This implementation uses CUDA kernels to process multiple pixels simultaneously, achieving significant speedup compared to CPU-based implementations.

## Features

- **CUDA-accelerated image processing**: Utilizes GPU parallelism for high-performance image filtering
- **Scalable design**: Efficiently handles images of various sizes, from small test images to high-resolution images
- **Configurable parameters**: Command-line arguments for kernel size and sigma (blur intensity)
- **PPM image format support**: Reads and writes P6 PPM format images
- **Performance metrics**: Reports processing time and throughput

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability 2.0 or higher)
- CUDA Toolkit (version 10.0 or higher)
- C++ compiler with C++11 support (g++, nvcc)
- Make build system

## Building the Project

### Prerequisites

1. Install CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Ensure `nvcc` (NVIDIA CUDA Compiler) is in your PATH
3. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

### Build Instructions

1. Clone or download this repository
2. Navigate to the project directory
3. Build the project using the provided Makefile:
   ```bash
   make
   ```

This will compile `cuda_image_processor.cu` and create the executable `cuda_image_processor`.

### Clean Build Artifacts

To remove compiled files:
```bash
make clean
```

## Usage

### Basic Usage

```bash
./cuda_image_processor <input_file> <output_file>
```

### Command Line Arguments

- `<input_file>`: Path to input PPM image file (P6 format)
- `<output_file>`: Path to output PPM image file
- `--kernel-size N`: Size of the Gaussian kernel (default: 15, must be odd number)
- `--sigma F`: Standard deviation for Gaussian distribution (default: 3.0)
- `--help`: Display help message

### Examples

1. **Basic blur with default parameters:**
   ```bash
   ./cuda_image_processor input.ppm output.ppm
   ```

2. **Custom kernel size and sigma:**
   ```bash
   ./cuda_image_processor input.ppm output.ppm --kernel-size 21 --sigma 5.0
   ```

3. **Light blur (small kernel, low sigma):**
   ```bash
   ./cuda_image_processor input.ppm output.ppm --kernel-size 9 --sigma 1.5
   ```

4. **Heavy blur (large kernel, high sigma):**
   ```bash
   ./cuda_image_processor input.ppm output.ppm --kernel-size 31 --sigma 8.0
   ```

## Image Format

The program supports PPM (Portable Pixmap) images in P6 format (binary RGB). To convert images from other formats:

- **Using ImageMagick:**
  ```bash
  convert input.jpg -format ppm output.ppm
  ```

- **Using FFmpeg:**
  ```bash
  ffmpeg -i input.jpg output.ppm
  ```

## Performance

The CUDA implementation provides significant performance improvements over CPU-based implementations, especially for large images. Performance scales with:
- Image resolution (more pixels = more parallelism)
- GPU compute capability
- Number of CUDA cores available

The program reports:
- Processing time in milliseconds
- Throughput in megapixels per second

## Code Structure

- `cuda_image_processor.cu`: Main source file containing:
  - Image I/O functions (ReadPPM, WritePPM)
  - CUDA kernel (GaussianBlurKernel)
  - Host code for memory management and kernel launching
  - Command-line argument parsing
  - Performance timing

- `Makefile`: Build configuration for compiling CUDA code

## Implementation Details

### CUDA Kernel Design

The Gaussian blur kernel uses a 2D thread block configuration (16x16 threads per block) to process image pixels in parallel. Each thread:
1. Computes its pixel coordinates from block and thread indices
2. Applies the Gaussian kernel to its pixel and neighboring pixels
3. Handles boundary conditions (edge pixels)
4. Writes the filtered result to output

### Memory Management

- Input and output images are allocated in device (GPU) memory
- Gaussian kernel is pre-computed on host and copied to device
- Efficient memory access patterns for optimal performance

### Boundary Handling

Edge pixels use clamping (repeating edge pixels) to handle kernel boundaries, ensuring the filter works correctly at image edges.

## Testing

To test the implementation:

1. Create or obtain a test PPM image
2. Run the processor:
   ```bash
   ./cuda_image_processor test_input.ppm test_output.ppm
   ```
3. Verify the output image shows the expected blur effect
4. Compare processing times for different image sizes

## Troubleshooting

- **"No CUDA devices found"**: Ensure you have an NVIDIA GPU and CUDA drivers installed
- **"Cannot open file"**: Verify the input file path is correct and the file exists
- **"Unsupported format"**: Ensure the input is a P6 PPM file
- **Compilation errors**: Verify CUDA Toolkit is properly installed and `nvcc` is accessible

## License

This project is created for educational purposes as part of the CUDA at Scale course.

## Author

Chikati Yatesh Chandra Sai

