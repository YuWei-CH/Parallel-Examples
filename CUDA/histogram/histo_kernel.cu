// Memory Coalescing (robust indexing)
__global__ void histo_kernel_MC(const char *buf, long size, unsigned int *histo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // stride = total # of threads
    int stride = blockDim.x * gridDim.x;

    // All threads in the gird collectively handle
    // blockDim.x * gridDim.x consecutive elements

    while (i < size)
    {
        unsigned char v = static_cast<unsigned char>(buf[i]);
        atomicAdd(&histo[v], 1u);
        i += stride;
    }
}

// Shared memory + Privatization (robust & corrected version)
__global__ void histo_kernel_SP(const char *buffer, long size, unsigned int *histo)
{
    // 1. Create private copies of the histo[] array for each thread block
    __shared__ unsigned int histo_private[256];

    // Initialize in strides so it works for any blockDim.x (>=1)
    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x)
    {
        histo_private[bin] = 0u;
    }
    __syncthreads();

    // 2. Use private copy to accumulate Grid–stride loop over the input
    long i = threadIdx.x + (long)blockIdx.x * blockDim.x;
    long stride = (long)blockDim.x * gridDim.x;

    while (i < size)
    {
        // Cast to unsigned to avoid negative indexing when char is signed
        unsigned char val = static_cast<unsigned char>(buffer[i]);
        atomicAdd(&histo_private[val], 1u);
        i += stride;
    }

    // 3. Merge block-private histograms into global histogram
    __syncthreads();
    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x)
    {
        atomicAdd(&histo[bin], histo_private[bin]);
    }
}

// Shared memory + Privatization with thread coarsening (2 items/thread/iteration)
__global__ void histo_kernel_SP_TC2(const char *buffer, long size, unsigned int *histo)
{
    // Private histogram per block in shared memory
    __shared__ unsigned int histo_private[256];

    // Initialize shared histogram (works for any blockDim.x)
    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x)
    {
        histo_private[bin] = 0u;
    }
    __syncthreads();

    // Grid–stride loop with coarsening factor 2
    long i = threadIdx.x + (long)blockIdx.x * blockDim.x;
    long stride = (long)blockDim.x * gridDim.x;

    while (i < size)
    {
        // 1st element
        unsigned char v0 = static_cast<unsigned char>(buffer[i]);
        atomicAdd(&histo_private[v0], 1u);

        // 2nd element (coarsened)
        long j = i + stride;
        if (j < size)
        {
            unsigned char v1 = static_cast<unsigned char>(buffer[j]);
            atomicAdd(&histo_private[v1], 1u);
        }

        i += 2 * stride;
    }

    // Merge to global histogram
    __syncthreads();
    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x)
    {
        atomicAdd(&histo[bin], histo_private[bin]);
    }
}