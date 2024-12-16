# Makefile for compiling main.cu with isa.h

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS =   # Adjust the architecture as needed

# Source files
SRC = main.cu

# Output executable
TARGET = main

# Default target
all: $(TARGET)

# Rule to compile the CUDA file
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean
