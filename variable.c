#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "malloc2D.h"

void function_level_OpenMP(int n, double *y)
{
    double *x;
    static double *x1;

    int thread_id;
#pragma omp parallel private(thread_id, x) shared(x1, n)
    {
        thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        if (thread_id == 0)
        {
            x = (double *)malloc(100 * n * sizeof(double));
            if (x == NULL)
            {
                fprintf(stderr, "Memory allocation failed for x\n");
                exit(EXIT_FAILURE);
            }
            printf("Thread %d: x is at address %p (private)\n", thread_id, (void *)x);
        }

        if (thread_id == 0)
        {
            x1 = (double *)malloc(100 * n * sizeof(double));
            if (x1 == NULL)
            {
                fprintf(stderr, "Memory allocation failed for x1\n");
                exit(EXIT_FAILURE);
            }
            printf("Thread %d: x1 is at address %p (shared static)\n", thread_id, (void *)x1);
        }

#pragma omp barrier // Wait for allocations to complete

        // Initialize x (private) if this thread allocated it
        if (thread_id == 0)
        {
            for (int i = 0; i < 100 * n; i++)
            {
                x[i] = i * 1.0;
            }
        }

// Initialize x1 (shared) collaboratively
#pragma omp for
        for (int i = 0; i < 100 * n; i++)
        {
            x1[i] = i * 2.0;
        }

#pragma omp barrier

        // Demonstrate private vs shared behavior
        printf("Thread %d accessing x at address %p\n", thread_id, (void *)x);
        // Only thread 0 can access x properly, other threads see NULL or random values
        if (thread_id == 0)
        {
            printf("Thread %d sees x[5] = %f (only thread 0 can properly access x)\n", thread_id, x[5]);
        }

        // All threads can access x1 since it's shared
        printf("Thread %d sees x1[5] = %f (all threads can access shared x1)\n", thread_id, x1[5]);

// Modify values to show how changes propagate
#pragma omp barrier

        if (thread_id == 0)
        {
            x[10] = thread_id + 100.0;
            printf("Thread %d changed x[10] to %f\n", thread_id, x[10]);
        }

        // Each thread modifies different elements of shared x1
        x1[thread_id + 20] = thread_id + 200.0;
        printf("Thread %d changed x1[%d] to %f\n", thread_id, thread_id + 20, x1[thread_id + 20]);

#pragma omp barrier

        // Show that changes to x1 are visible to all threads
        if (thread_id == 0)
        {
            printf("\nFinal values of modified x1 elements:\n");
            for (int i = 0; i < num_threads; i++)
            {
                printf("x1[%d] = %f\n", i + 20, x1[i + 20]);
            }
        }

        if (thread_id == 0)
        {
            free(x);
        }
        if (thread_id == 0)
        {
            free(x1);
        }
    }
}

int main(int argc, char **argv)
{
    int n = 10;
    double *y = (double *)malloc(n * sizeof(double));

    printf("\n=== Demonstrating Private vs Shared Variables in OpenMP ===\n");
    printf("x: private pointer variable (each thread has its own copy of pointer)\n");
    printf("x1: shared static pointer variable (all threads see the same pointer)\n\n");

    function_level_OpenMP(n, y);

    printf("\nSummary:\n");
    printf("- Private variables (x): Each thread has its own copy of the variable\n");
    printf("- Shared variables (x1): All threads access the same memory location\n");
    printf("- In this example, only thread 0 allocates memory for both x and x1\n");
    printf("- However, since x is private, only thread 0 can access that memory\n");
    printf("- Since x1 is shared (static), all threads can access the memory it points to\n");

    free(y);
    return 0;
}