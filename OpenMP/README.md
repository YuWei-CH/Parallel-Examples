# OpenMP (OMP) Examples

OpenMP (Open Multi-Processing) is an API that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It provides a simple and flexible interface for developing parallel applications on shared memory architectures.

## OpenMP environment variable
```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=true
# if finetune perf
export OMP_NUM_THREADS=6
```

## Compiling OpenMP Programs

All OpenMP programs can be compiled using:

```bash
# With GCC
gcc -fopenmp program.c -o program

# With CMake (using provided CMakeLists.txt)
mkdir build && cd build
cmake ..
make
```

## Hello OpenMP

Simple demonstration of OpenMP parallel region:

```bash
gcc -fopenmp HelloOpenMP.c -o hello_openmp
./hello_openmp
```

## Variable Scoping in OpenMP

Demonstrates private vs shared variables in OpenMP:

```bash
gcc -fopenmp variable.c malloc2D.c -o variable
./variable
```

This example shows how different variable scopes (private, shared, etc.) behave in parallel regions.

## Vector Addition Examples

Several vector addition implementations with different optimizations:

- `vecadd_opt.c`: Serial implementation
- `vecadd_opt1.c`: Basic OpenMP parallelization
- `vecadd_opt2.c`: OpenMP with first-touch optimization (better NUMA performance)

```bash
gcc -fopenmp vecadd_opt.c timer.c -o vecadd_opt
gcc -fopenmp vecadd_opt1.c timer.c -o vecadd_opt1
gcc -fopenmp vecadd_opt2.c timer.c -o vecadd_opt2
```

Run and compare performance:

```bash
./vecadd_opt
./vecadd_opt1
./vecadd_opt2
```

## Understanding Stencil Computations

Stencil computations are a pattern commonly used in numerical methods for solving partial differential equations, image processing, and cellular automata. A stencil defines a fixed pattern that is applied to each element in a grid, where each output value depends on the corresponding input value and its neighbors.

### What is a Stencil?

In a stencil operation, each point in a multi-dimensional grid is updated with a weighted sum of neighboring points. For example, in a 5-point stencil (commonly used in 2D heat diffusion):

```
                    ┌───┐
                    │ N │
                    └───┘
            ┌───┐   ┌───┐   ┌───┐
            │ W │   │ C │   │ E │
            └───┘   └───┘   └───┘
                    ┌───┐
                    │ S │
                    └───┘
```

Here, the new value at the Center point is computed as a function of its North, South, East, West neighbors.

### Stencil Examples (stencil_opt*.c)

The repository contains multiple stencil implementations with different optimization techniques:

- `stencil_opt2.c`: Basic parallelized stencil
- `stencil_opt3.c`: Optimized with nowait clause
- `stencil_opt4.c`: Using persistent parallel regions
- `stencil_opt6.c`: Advanced optimization with manual work distribution

Compilation:

```bash
gcc -fopenmp stencil_opt2.c malloc2D.c timer.c -o stencil_opt2
gcc -fopenmp stencil_opt3.c malloc2D.c timer.c -o stencil_opt3
gcc -fopenmp stencil_opt4.c malloc2D.c timer.c -o stencil_opt4
gcc -fopenmp stencil_opt6.c malloc2D.c timer.c -o stencil_opt6
```

Run and compare performance:

```bash
./stencil_opt2
./stencil_opt3
./stencil_opt4
./stencil_opt6
```

Each version demonstrates different optimization techniques:
- Basic parallelization with OpenMP directives
- Memory access pattern optimization
- Reducing synchronization overhead
- Thread affinity and data locality improvements
- Manual work distribution for better load balancing

## Reduction Example

Simple example showing reduction operations in OpenMP:

```bash
# This is part of a larger file in the examples
gcc -fopenmp -c serial_sum_novec.c

# The function demonstrates parallel reduction:
# double do_sum_novec(double *restrict var, long ncells)
# {
#     double sum = 0.0;
#     #pragma omp parallel for reduction(+ : sum)
#     for (long i = 0; i < ncells; ++i)
#     {
#         sum += var[i];
#     }
#     return sum;
# }
```

## Performance Tips

For best OpenMP performance:
- Set thread affinity correctly with `OMP_PLACES=cores` and `OMP_PROC_BIND=true`
- Adjust thread count to match your CPU with `OMP_NUM_THREADS=N`
- Consider NUMA effects in memory allocation (first-touch policy)
- Minimize synchronization points between parallel regions
- Balance workload among threads for optimal scaling