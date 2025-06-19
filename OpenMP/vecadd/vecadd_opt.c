#include <stdio.h>
#include <time.h>
#include "timer.h"

#define ARRAY_SIZE 80000000
static double a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

void vector_add(double *c,
                double *a,
                double *b,
                int n);

int main(int argc, char *argv[])
{
    printf("Using single thread");
    struct timespec tstart;
    double time_sum = 0.0;
    // Loop Initialization
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    cpu_timer_start(&tstart);
    vector_add(c, a, b, ARRAY_SIZE);
    time_sum += cpu_timer_stop(tstart);

    printf("Time taken for vector addition: %f seconds\n", time_sum);
}

void vector_add(double *c, double *a, double *b, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}