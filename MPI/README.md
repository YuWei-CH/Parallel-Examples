# MPI Examples

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
Runs on 8 PEs and does a "circular shift." This means that every PE sends
some data to its nearest neighbor either "up" (one PE higher) or "down." To make it circular,
PE 7 and PE 0 are treated as neighbors.
```bash
mpicc -o circular_shift circular_shift.c
# use --use-hwthread-cpus to recognize Hyper-threading
mpirun --use-hwthread-cpus -np 8 ./circular_shift
```

## My_MPI_Comm_size()
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