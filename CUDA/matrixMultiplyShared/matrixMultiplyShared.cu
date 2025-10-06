#define TILE_WIDTH 16

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory
    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    for (unsigned int tile = 0; tile < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; ++tile)
    {
        // Load tile to A_s
        if (row < numARows && tile * TILE_WIDTH + threadIdx.x < numAColumns)
        {
            A_s[threadIdx.y][threadIdx.x] = A[row * numAColumns + tile * TILE_WIDTH + threadIdx.x];
        }
        else
        {
            A_s[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile into B_S
        if (col < numBColumns && tile * TILE_WIDTH + threadIdx.y < numBRows)
        {
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_WIDTH + threadIdx.y) * numBColumns + col];
        }
        else
        {
            B_s[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads;

        // compute tile
        for (unsigned int i = 0; i < TILE_WIDTH; ++i)
        {
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads;
    }
    if (row < numCRows && col < numCColumns)
    {
        C[row * numCColumns + col] = sum;
    }
}