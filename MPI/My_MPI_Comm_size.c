#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int my_PE_num, numberofnodes, data;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);

    if (my_PE_num == 0)
    {
        for (numberofnodes = 1; numberofnodes < 10; numberofnodes++)
        {
            if (MPI_Send(&data, 1, MPI_INT, numberofnodes, 10, MPI_COMM_WORLD) != MPI_SUCCESS)
            {
                break;
            }
        }
    }
    printf("The number of nodes is %d.", numberofnodes);
    MPI_Finalize();
}