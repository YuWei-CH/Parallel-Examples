#define MASK_WIDTH 5

// basic version
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *O, unsigned int mask_width, unsigned int width)
{
    // 1d
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int start_index = i - (mask_width / 2);

    float p_sum = 0.0f;
    for (size_t m = 0; m < mask_width; ++m)
    {
        if ((start_index + m) >= 0 && (start_index + m) < width)
        {
            p_sum += N[start_index + m] * M[m];
        }
    }
    O[i] = p_sum;
}

// Using constant memory

// Constant Mem
__constant__ float Mc[MASK_WIDTH];

__global__ void convolution_1D_basic_kernel_Mc(float *N, float *O, unsigned int width)
{
    // 1d
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int start_index = i - (MASK_WIDTH / 2);

    float p_sum = 0.0f;
    for (size_t m = 0; m < MASK_WIDTH; ++m)
    {
        if ((start_index + m) >= 0 && (start_index + m) < width)
        {
            p_sum += N[start_index + m] * Mc[m];
        }
    }
    O[i] = p_sum;
}