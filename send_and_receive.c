#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int my_PE_num, numtoreceive, numtosend = 42;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);

    if (my_PE_num == 0)
    {
        MPI_Recv(&numtoreceive, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // MPI_ANY_SOURCE: receive from any source
        printf("Number received is: %d\n", numtoreceive);
    }
    else
        MPI_Send(&numtosend, 1, MPI_INT, 0, 10, MPI_COMM_WORLD); // 0 is the destination PE number, 10 is the tag

    MPI_Finalize();
}