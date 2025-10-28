# Conv2D Benchmark (Pure CUDA vs cuDNN vs PyTorch)

This folder contains three equivalent 2D convolution implementations so you can explore performance differences:

1. `conv2d_cuda.cu` – hand-written CUDA kernel (baseline implementation).
2. `conv2d_cudnn.cu` – cuDNN-based version using the fastest algorithm cuDNN suggests.
3. `conv2d_pytorch.py` – PyTorch module, which internally selects the best cuDNN kernel.

Each program uses identical tensor shapes, strides, padding, deterministic inputs, and weights. They all run a few warm-up iterations to amortize lazy initialization and then time 50 back-to-back forward passes on the GPU, reporting:

- Total kernel time across the timed iterations.
- Average kernel time per iteration.
- Output checksum (used to confirm numerical parity across implementations).

The current benchmark configuration is sized to exercise an RTX 3060 12 GB:

```
batch_size = 8
in_channels = 64
in_height = in_width = 512
out_channels = 128
kernel_size = 3
stride = 1
padding = 1
warmup_iterations = 5
timed_iterations = 50
```

---

## Prerequisites
- cuDNN headers and libs installed (e.g., `/usr/include/x86_64-linux-gnu`, `/usr/lib/x86_64-linux-gnu`).
- Python 3 with PyTorch built against CUDA/cuDNN.

---

## Build & Run

### Pure CUDA baseline (`conv2d_cuda.cu`)

```bash
nvcc CUDA/DeepLearning/conv2d_cuda.cu -o CUDA/DeepLearning/conv2d_cuda
./CUDA/DeepLearning/conv2d_cuda
```

### cuDNN implementation (`conv2d_cudnn.cu`)

```bash
nvcc CUDA/DeepLearning/conv2d_cudnn.cu -o CUDA/DeepLearning/conv2d_cudnn \
    -lcudnn -I/usr/include/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu
./CUDA/DeepLearning/conv2d_cudnn
```

If cuDNN lives elsewhere, replace the include/library paths accordingly.

### PyTorch implementation (`conv2d_pytorch.py`)

```bash
python CUDA/DeepLearning/conv2d_pytorch.py
```

All three executables/scripts print the output tensor shape, total and average kernel time, and a checksum. Matching checksums confirm that every implementation is convolving the same data with the same parameters.

---

## How the Comparison Works

1. **Deterministic data** – Inputs follow `(i % 255) / 255.0f`, weights use `(i % 10) / 10.0f`, biases are zeroed. This makes the outputs byte-for-byte comparable.  
2. **Warm-up iterations** – Five unmeasured runs prime CUDA/cuDNN/PyTorch so that lazy kernel compilation and allocator overheads do not skew the results.  
3. **Timed loop** – Fifty forward passes run back-to-back on the existing device buffers. CUDA events (or PyTorch `cuda.Event`) capture GPU-only time; host-device transfers are excluded.  
4. **Reporting** – Each program prints the total milliseconds across the 50 iterations, the per-iteration average, and the checksum.  

With this setup on an RTX 3060 12 GB you should observe:

- The custom CUDA kernel is functionally correct but ~20× slower because it lacks shared-memory tiling and tensor-core optimizations.  
- The cuDNN program, once allowed to pick its fastest algorithm, delivers performance close to PyTorch.  
- PyTorch serves as the reference implementation, effectively wrapping cuDNN with the same inputs/weights.
