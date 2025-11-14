# Makefile for CUDA Image Processor

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_60 -std=c++11 -Xcompiler -Wall

# Target executable
TARGET = cuda_image_processor

# Source files
SRC = cuda_image_processor.cu

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Clean target
clean:
	rm -f $(TARGET) *.o

# Phony targets
.PHONY: all clean

