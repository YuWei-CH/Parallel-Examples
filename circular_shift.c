#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int my_PE_num, num_to_send, message_received;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);

    num_to_send = my_PE_num;

    /*
        To avoid deadlock, we need to make sure that the send and receive operations are in the correct order.
    */
    if (my_PE_num == 7)
    {
        MPI_Recv(&message_received, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status); // Step 1: Recieve from PE 0
        MPI_Ssend(&num_to_send, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);                           // Step 2: Send to PE 0
    }
    else
    {
        MPI_Ssend(&num_to_send, 1, MPI_INT, my_PE_num + 1, 10, MPI_COMM_WORLD);               // Step 1: Send to next PE
        MPI_Recv(&message_received, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status); // Step 2: Recieve from previous PE
    }
    printf("PE %d received %d.\n", my_PE_num, message_received);
    MPI_Finalize();
}