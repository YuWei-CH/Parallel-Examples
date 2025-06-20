#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "malloc2D.h"
#include "timer.h"

// Swap
#define SWAP_PTR(xnew, xold, xtmp) (xtmp = xnew, xnew = xold, xold = xtmp)

int main(int argc, char *argv[])
{
#pragma omp parallel
#pragma omp master
    printf("Number of threads: %d\n", omp_get_num_threads());

    struct timespec tstart_init, tstart_flush, tstart_stencil, tstart_total;
    double init_time, flush_time, stencil_time, total_time;
    int imax = 2002, jmax = 2002;
    double **xtmp;
    double **x = malloc2D(jmax, imax);
    double **xnew = malloc2D(jmax, imax);
    int *flush = (int *)malloc(jmax * imax * sizeof(int) * 4);

    cpu_timer_start(&tstart_total);
    cpu_timer_start(&tstart_init);
#pragma omp parallel for
    for (size_t j = 0; j < jmax; ++j)
    {
        for (size_t i = 0; i < imax; ++i)
        {
            xnew[j][i] = 0.0;
            x[j][i] = 5.0;
        }
    }

#pragma omp parallel for
    for (size_t j = jmax / 2 - 5; j < jmax / 2 + 5; ++j)
    {
        for (size_t i = imax / 2 - 5; i < imax / 2 - 1; ++i)
        {
            x[j][i] = 400.0;
        }
    }
    init_time += cpu_timer_stop(tstart_init);

    for (size_t iter = 0; iter < 10000; ++iter)
    {
        cpu_timer_start(&tstart_flush);
#pragma omp parallel for
        for (size_t l = 1; l < jmax * imax * 4; ++l)
        {
            flush[l] = 1.0; // Force flush to mimic code cache: code does not have var from prior operation.
        }
        flush_time += cpu_timer_stop(tstart_flush);
        cpu_timer_start(&tstart_stencil);
#pragma omp parallel for
        for (size_t j = 1; j < jmax - 1; ++j)
        {
            for (size_t i = 1; i < imax - 1; ++i)
            {
                xnew[j][i] = (x[j][i] + x[j][i - 1] + x[j][i + 1] +
                              x[j - 1][i] + x[j + 1][i]) /
                             5.0;
            }
        }
        stencil_time += cpu_timer_stop(tstart_stencil);

        SWAP_PTR(xnew, x, xtmp);
        if (iter % 1000 == 0)
            printf("Iteration %d completed.\n", (int)iter);
    }
    total_time = cpu_timer_stop(tstart_total);

    printf("Timing: init %f flush %f stencil %f total %f\n",
           init_time, flush_time, stencil_time, total_time);

    free(x);
    free(xnew);
    free(flush);
}
