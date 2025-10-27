# CUDA Examples

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model for GPUs.

## Native CUDA Examples

This repo includes CUDA C++ examples in `CUDA/`:

- `vecAdd/vecAdd.cu`: Vector addition using 1D grid/block launch
- `matrixMultiply/matrixMultiply.cu`: Dense matrix multiplication (naive, no shared memory)
- `matrixMultiplyShared/matrixMultiplyShared.cu`: Matrix multiplication with shared-memory tiling (TILE_WIDTH=16)
- `convKernel/1dConvKernel.cu`: 1D convolution (basic global-memory version + constant memory mask variant)
- `convKernel/tiled1dConvKernel.cu`: 1D convolution with two shared-memory tiling strategies (halo loading vs on-demand hybrid)

- `Histogram/histo_kernel.cu` + `Histogram/histo_host.cu`: GPU histogram (8-bit, 256 bins) with three variants:
      - MC: memory-coalesced global atomics
      - SP: shared-memory privatization per block
      - SP_TC2: shared-memory + thread coarsening (2 items/thread/iteration)

### Data Layout

Each program reads simple text files from its local `data/` subfolder (shared-memory version uses the same format as the naive one):

- Vector add: `input1.txt`, `input2.txt` (one float per whitespace) -> writes `output.txt`
- Matrix multiply: `A.txt`, `B.txt` each start with: `rows cols` on the first line, followed by all elements row-major; output written to `C.txt` with the same header format.

### Quick Compile & Run (without libwb)

```bash
# From repository root
cd CUDA/vecAdd
nvcc vecAdd.cu -o vecAdd
./vecAdd

cd ../matrixMultiply
nvcc matrixMultiply.cu -o matrixMultiply
./matrixMultiply

# Shared-memory tiled version (same input data format)
cd ../matrixMultiplyShared
nvcc matrixMultiplyShared.cu -o matrixMultiplyShared
./matrixMultiplyShared

# 1D Convolution kernels (adjust file names / add host driver as needed)
cd ../../convKernel
nvcc 1dConvKernel.cu -o conv1d_basic
nvcc tiled1dConvKernel.cu -o conv1d_tiled

# Histogram kernels (MC, SP, SP_TC2)
cd ../Histogram
make

# Run with default random input (64 MB)
./histo_host

# Run with specific size (e.g., 16 MB)
./histo_host 16777216

# Or run with a binary input file of bytes (0-255)
./histo_host data/your_bytes.bin
```

### Hardware Note (RTX 30 Series GPU)

Parameters like `blockDim = 512` in `vecAdd.cu` and `TILE_WIDTH = 16` in `matrixMultiply.cu` were chosen to run reasonably well on a typical NVIDIA RTX 30 (Ampere) GPU. They are simple, readable defaults rather than fully optimized values.