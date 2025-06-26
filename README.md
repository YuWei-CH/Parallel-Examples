# Parallel-Examples
MPI examples
OpenMP(OMP) examples
CUDA & PyCUDA examples


## Amdahl’s Law

Amdahl’s Law calculates the **speedup (S)** of a parallel program compared to its serial execution:

$S = \frac{1}{(1 - P) + \frac{P}{N}}$

Where:
- **S** = Speedup (how many times faster the parallel execution is compared to serial)
- **P** = The portion of the program that **can be parallelized** (as a percentage)
- **1 - P** = The portion that **must remain serial** (cannot be parallelized)
- **N** = Number of processing units (e.g., CPU cores, GPU threads)

This formula shows that the **speedup is limited by the serial portion of the program**. Even with an infinite number of processing units, the maximum achievable speedup is:

$S = \frac{1}{(1 - P) + \frac{P}{N}}$

which means the **serial portion becomes the bottleneck** as N increases.

## Hello World
Print Hello World from each processes
```bash
# Compile
mpicc -o hello_world hello_world.c
# Run with 8 process
mpirun -np 8 ./hello_world
```

## Send_and_Receive
It will run on 2 PEs and will send a simple message (the number 42) from PE 1 to PE 0. PE 0 will then print this out.
```bash
mpicc -o send_and_receive send_and_receive.c
mpirun -np 2 ./send_and_receive
```

## Synchronization
Our code will perform the rather pointless operations of:
1. Have PE 0 send a number to the other 3 PEs
2. have them multiply that number by their own PE number
3. they will then print the results out, in order
4. and send them back to PE 0
5. which will print out the sum.
```bash
mpicc -o synchronization synchronization.c
mpirun -np 4 ./synchronization
```

## Finding Pi
Using numerical integration based on the Midpoint Rule for approximating integrals to calculate PI.
```bash
mpicc -o finding_pi finding_pi.c -lm
mpirun -np 6 ./finding_pi
```

## Circular Shift
Runs on 8 PEs and does a “circular shift.” This means that every PE sends
some data to its nearest neighbor either “up” (one PE higher) or “down.” To make it circular,
PE 7 and PE 0 are treated as neighbors.
```bash
mpicc -o circular_shift circular_shift.c
# use --use-hwthread-cpus to recognize Hyper-threading
mpirun --use-hwthread-cpus -np 8 ./circular_shift
```

##  My_MPI_Comm_size()
Implement MPI_Comm_size() only using MPI_Init,
MPI_Comm_Rank, MPI_Send, MPI_Recv, MPI_Barrier, MPI_Finalize
**Honestly, the solution 100% work isn't exist.**
```bash
mpicc -o My_MPI_Comm_size My_MPI_Comm_size.c
mpirun -np 6 ./My_MPI_Comm_size
```
This should not work, because numberofnodes not initial at PE 1-9

## Laplace Solver
serial_laplace_solver
```bash
gcc -o serial_laplace_solver serial_laplace_solver.c -lm
./serial_laplace_solver
## Max error at iteration 18477 was 0.001000
## Total time was 99.293923 seconds.
```

Parallel Version
```bash
/****************************************************************
 * Laplace MPI C Version
 *
 * T is initially 0.0
 * Boundaries are as follows
 *
 *                T                      4 sub-grids
 *   0  +-------------------+  0    +-------------------+
 *      |                   |       |                   |
 *      |                   |       |-------------------|
 *      |                   |       |                   |
 *   T  |                   |  T    |-------------------|
 *      |                   |       |                   |
 *      |                   |       |-------------------|
 *      |                   |       |                   |
 *   0  +-------------------+ 100   +-------------------+
 *      0         T       100
 *
 * Each PE only has a local subgrid.
 * Each PE works on a sub grid and then sends
 * its boundaries to neighbors.
 *
 * John Urbanic, PSC 2014
 *
 *******************************************************************/
 ```
```bash
mpicc -o parallel_laplace_solver parallel_laplace_solver.c -lm
mpirun -np 4 ./parallel_laplace_solver
## Max error at iteration 29182 was 0.001000
## Total time was 41.817048 seconds.
```

## Debug and Profiling
GDB
```bash
mpicc -o synchronization synchronization.c
mpirun -np 4 gdb ./synchronization
## If GUI
mpirun -np 4 xterm -e gdb ./synchronization
```

# OpenMP(OMP)

## OpenMP environment variable
```bash
export OMP_PLACES=cores
export OMP_PROC_BIND=true
# if finetune perf
export OMP_NUM_THREADS=6
```

# OpenMP Examples

OpenMP (Open Multi-Processing) is an API that supports multi-platform shared memory multiprocessing programming in C, C++, and Fortran. It provides a simple and flexible interface for developing parallel applications on shared memory architectures.

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

# CUDA & PyCUDA

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model for GPUs. PyCUDA provides Python bindings for CUDA, allowing you to write GPU-accelerated code in Python.

## Setup Virtual Environment and Install PyCUDA

Create a virtual environment for Python CUDA development:

```bash
# Create virtual environment
python3 -m venv cuda_env

# Activate virtual environment
source cuda_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyCUDA
pip install pycuda
```

To deactivate the virtual environment when done:

```bash
deactivate
```

**Note**: PyCUDA requires NVIDIA CUDA toolkit to be installed on your system. Make sure you have compatible NVIDIA drivers and CUDA toolkit installed before installing PyCUDA.