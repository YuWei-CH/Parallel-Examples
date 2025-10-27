#define MASK_WIDTH 5
#define TILE_WIDTH 256
#define RADIUS (MASK_WIDTH / 2)

__constant__ float Mc[MASK_WIDTH];

// Strategy 1
__global__ void convolution_1D_strategy1_kernel(float *N, float *O, unsigned int width)
{
    // SMEM for input + halo
    __shared__ float N_shared[TILE_WIDTH + MASK_WIDTH - 1];

    // 1d - TILE_WIDTH
    int i = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Left halo and Right halo
    int halo_left_idx = blockIdx.x * TILE_WIDTH - RADIUS; // e.g. 2 items before this tile
    int halo_right_idx = (blockIdx.x + 1) * TILE_WIDTH;   // e.g. 2 items after this tile (== first 2 items of next tile)

    // Step1: Load left halo
    if (threadIdx.x < RADIUS)
    {
        int load_idx = halo_left_idx + threadIdx.x;
        N_shared[threadIdx.x] = (load_idx >= 0 && load_idx < width) ? N[load_idx] : 0.0f;
    }

    // Step 2: Load core(output length) data
    if (i < width)
    {
        N_shared[threadIdx.x + RADIUS] = N[i];
    }
    else // More than width
    {
        N_shared[threadIdx.x + RADIUS] = 0.0f;
    }

    // Step 3: Load right halo
    if (threadIdx.x >= TILE_WIDTH - RADIUS) // >= 254
    {
        int load_idx = halo_right_idx + (threadIdx.x - (TILE_WIDTH - RADIUS));                             // halo_right_idx + 255 - (256 - 2)  = next tile 1 or 2
        N_shared[threadIdx.x + MASK_WIDTH - 1] = (load_idx >= 0 && load_idx < width) ? N[load_idx] : 0.0f; //  N_shared[255+4] or N_shared[256+4]
    }

    __syncthreads();

    // Compute
    // Only output thread compute, halo will not compute
    if (threadIdx.x >= RADIUS && threadIdx.x < TILE_WIDTH - RADIUS && i < width)
    {
        float p_sum = 0.0f;
        for (size_t m = 0; m < MASK_WIDTH; ++m)
        {
            p_sum += N_shared[threadIdx.x + m - RADIUS] * Mc[m];
        }
        O[i] = p_sum;
    }
}

// Strategy 3
__global__ void convolution_1D_strategy3_kernel(float *N, float *O, unsigned int width)
{
    __shared__ float N_shared[TILE_WIDTH]; // No halo

    int i = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Only load data within the TILE size
    if (i < width)
    {
        N_shared[threadIdx.x] = N[i];
    }
    else
    {
        N_shared[threadIdx.x] = 0.0f;
    }

    // Compute (only tile thread involve)
    if (i < width)
    {
        float p_sum = 0.0f;
        for (size_t m = 0; m < MASK_WIDTH; ++m)
        {
            int global_idx = i + m - RADIUS; // Include halo, here is all items we need
            if (global_idx >= 0 && global_idx < width)
            {                                                                               // valid boundary
                if (threadIdx.x + m - RADIUS >= 0 && threadIdx.x + m - RADIUS < TILE_WIDTH) // inside tile, read from tile
                {
                    p_sum += N_shared[threadIdx.x + m - RADIUS] * Mc[m];
                }
                else
                { // Halo outside tile, read from global mem
                    p_sum += N[global_idx] * Mc[m];
                }
            } // Ignore, or fill with 0.0
        }
        O[i] = p_sum;
    }
}