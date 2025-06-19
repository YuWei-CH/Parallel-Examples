#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int my_PE_num, numtoreceive, numtosend = 4, index, result = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num); // Get the rank of the current node

    if (my_PE_num == 0)
    {
        for (index = 1; index < 4; index++) // Start from node 1 because node 0 is the master node
        {
            MPI_Send(&numtosend, 1, MPI_INT, index, 10, MPI_COMM_WORLD); // 1. Have PE 0 send a number to the other 3 PEs
        }
    }
    else
    {
        MPI_Recv(&numtoreceive, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status); // Receive numtosend from master node(0)
        result = numtoreceive * my_PE_num;                                   // 2. have them multiply that number by their own PE number
    }
    for (index = 1; index < 4; index++)
    { /*
        Synchronize all nodes, ensure that all
        processes enter the next round of the loop
        (that's why it will print in order)
        at the same moment to avoid processes
        executing out of order.
      */
        MPI_Barrier(MPI_COMM_WORLD);
        if (index == my_PE_num)
            printf("PE %d's result is: %d.\n", my_PE_num, result); // 3. they will then print the results out, in order
    }
    if (my_PE_num == 0)
    {
        for (index = 1; index < 4; index++)
        {
            MPI_Recv(&numtoreceive, 1, MPI_INT, index, 10, MPI_COMM_WORLD, &status); // Receive the results from the other 3 PEs
            result += numtoreceive;                                                  // 5. which will print out the sum.
        }
        printf("Total is %d.\n", result); // Print out the total, not in order
    }
    else
    {
        MPI_Send(&result, 1, MPI_INT, 0, 10, MPI_COMM_WORLD); // 4. and send them back to PE 0
    }
    MPI_Finalize();
}