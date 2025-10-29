# Simple Makefile to build histogram kernels and host runner

CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
ARCH ?= -arch=sm_70
CXXFLAGS ?= -O3 -std=c++17 $(ARCH)
LDFLAGS ?= 

TARGET := histo_host
SRCS := histo_host.cu histo_kernel.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
