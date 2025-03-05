# MPI-Examples
MPI examples

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