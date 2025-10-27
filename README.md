# Parallel-Examples

## Purpose
This repo include Parallel Tutorial/Workshop examples:
Point to each's readme.# Parallel Computing Examples

This repository contains hands-on examples for three major parallel computing technologies: MPI, OpenMP, and CUDA. Each technology folder includes practical implementations with detailed documentation.

## Technologies Covered

### [MPI (Message Passing Interface)](MPI/README.md)
- Distributed memory parallel computing
- Examples include basic communication patterns, synchronization, and a parallel Laplace solver
- Best for multi-node cluster computing

### [OpenMP (Open Multi-Processing)](OpenMP/README.md)
- Shared memory parallel computing for multi-core CPUs
- Examples include vector addition, stencil computations, and reduction operations
- Best for single-node multi-core systems

### [CUDA](CUDA/README.md)
- GPU parallel computing platform by NVIDIA
- Examples include vector addition, matrix multiplication, convolution, and histogram computation
- Best for data-parallel computations on compatible NVIDIA GPUs

## Key Concept

### Amdahl's Law

Amdahl's Law calculates the **speedup (S)** of a parallel program compared to its serial execution:

$S = \frac{1}{(1 - P) + \frac{P}{N}}$

Where:
- **S** = Speedup (how many times faster the parallel execution is compared to serial)
- **P** = The portion of the program that **can be parallelized** (as a percentage)
- **1 - P** = The portion that **must remain serial** (cannot be parallelized)
- **N** = Number of processing units (e.g., CPU cores, GPU threads)

This formula shows that the **speedup is limited by the serial portion of the program**. Even with an infinite number of processing units, the maximum achievable speedup is:

$\lim_{N\to\infty} S = \frac{1}{1 - P}$

which means the **serial portion becomes the bottleneck** as N increases.

## References & Resources

### MPI Resources
- [PSC MPI Workshop](https://www.psc.edu/)
- [MPI Tutorial](https://mpitutorial.com/tutorials/)
- *Parallel and High Performance Computing* by Robert Robey and Yuliana Zamora

### OpenMP Resources
- *Parallel and High Performance Computing* by Robert Robey and Yuliana Zamora

### CUDA Resources
- *Programming Massively Parallel Processors (PMPP)*
- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## Contribution
Feel free to PR if you want to add more examples. 
Examples are tested with:
OS: Ubuntu 22.04 LTS
CPU: AMD rx5600x
GPU: Nvidia RTX 3060 (Ampere Arch)