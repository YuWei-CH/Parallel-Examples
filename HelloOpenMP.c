#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
#pragma omp parallel
    {
        int nthreads, thread_id;
        nthreads = omp_get_num_threads(); // Get the number of threads in the current team
        thread_id = omp_get_thread_num(); // Get the ID of the current thread
#pragma omp master
        {
            printf("Goodbye slow serial world! and hello OpenMP!\n");
            printf("Hello OpenMP! I have %d thread(s) available and my thread id is %d\n", nthreads, thread_id);
        }
    }
}