// reduction_kernels.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <numeric> // std::accumulate
#include <cmath>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ------------------------------
// Helpers
// ------------------------------
__inline__ __device__ float warpReduceSum(float v)
{
    // Reduce within a warp using shuffles (no shared memory, no __syncthreads)
    for (int ofs = WARP_SIZE >> 1; ofs > 0; ofs >>= 1)
        v += __shfl_down_sync(0xffffffff, v, ofs);
    return v;
}

// Block-wide reduction that uses shared memory until one warp remains,
// then switches to warp shuffles to avoid extra barriers.
template <int BLOCK_DIM>
__inline__ __device__ float blockReduceSum(float v, float *__restrict__ smem)
{
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();

    // Reduce in shared memory until <= 32 threads left
    for (int stride = BLOCK_DIM >> 1; stride >= WARP_SIZE; stride >>= 1)
    {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    // Final warp: move partial to register and finish with shuffles
    float sum = smem[tid];
    if (tid < WARP_SIZE)
    {
        sum = warpReduceSum(sum);
    }
    return sum; // only lane 0 holds the block sum after this
}

// ------------------------------
// 1) Basic Reduction Tree (simple, pedagogical)
//    - One block reduces a 2*BLOCK_DIM slice
//    - Interleaved addressing pattern; illustrative but not optimal
// ------------------------------
template <int BLOCK_DIM>
__global__ void reduce_basic(const float *__restrict__ in, int N,
                             float *__restrict__ block_sums)
{
    __shared__ float sdata[BLOCK_DIM];

    // Each block handles a 2*BLOCK_DIM chunk
    const int tid = threadIdx.x;
    const int base = 2 * BLOCK_DIM * blockIdx.x;

    // Load two elements per thread when available
    float x = 0.0f;
    int i0 = base + tid;
    int i1 = base + tid + BLOCK_DIM;
    if (i0 < N)
        x += in[i0];
    if (i1 < N)
        x += in[i1];

    sdata[tid] = x;
    __syncthreads();

    // "Textbook" tree: stride doubles (interleaved addressing)
    // This version may suffer control divergence and bank/bandwidth inefficiency.
    for (int stride = 1; stride < BLOCK_DIM; stride <<= 1)
    {
        int idx = 2 * stride * tid;
        if (idx + stride < BLOCK_DIM)
        {
            sdata[idx] += sdata[idx + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        block_sums[blockIdx.x] = sdata[0];
}

// ------------------------------
// 2) Reduction Tree addressing control & memory divergence
//    - Coalesced loads (2 per thread)
//    - Sequential addressing (stride halves)
//    - Warp-shuffle tail to cut barriers
// ------------------------------
template <int BLOCK_DIM>
__global__ void reduce_optimized(const float *__restrict__ in, int N,
                                 float *__restrict__ block_sums)
{
    __shared__ float sdata[BLOCK_DIM];

    const int tid = threadIdx.x;
    const int base = 2 * BLOCK_DIM * blockIdx.x;

    // Coalesced, 2 elements per thread
    float x = 0.0f;
    int i0 = base + tid;
    int i1 = base + tid + BLOCK_DIM;
    if (i0 < N)
        x += in[i0];
    if (i1 < N)
        x += in[i1];

    sdata[tid] = x;
    __syncthreads();

    // Sequential addressing (stride halves): reduces control divergence
    for (int stride = BLOCK_DIM >> 1; stride >= WARP_SIZE; stride >>= 1)
    {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Warp tail via shuffles
    float sum = sdata[tid];
    if (tid < WARP_SIZE)
        sum = warpReduceSum(sum);
    if (tid == 0)
        block_sums[blockIdx.x] = sum;
}

// ------------------------------
// 3) Segmented Reduction with thread coarsening + optimized tree
//    - Grid-stride loop with COARSE_FACTOR: each thread accumulates multiple elems in registers
//    - Per-block reduction uses warp-shuffle tail to minimize barriers
//    - Each block writes one partial; caller can either atomicAdd to a global sum
//      or launch a second kernel to reduce block_sums
// ------------------------------
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

template <int BLOCK_DIM>
__global__ void reduce_segmented_coarsened(const float *__restrict__ in, int N,
                                           float *__restrict__ block_sums)
{
    __shared__ float sdata[BLOCK_DIM];

    const int tid = threadIdx.x;
    const int totalThreads = BLOCK_DIM * gridDim.x; // all threads in the grid
    const int gtid = blockIdx.x * BLOCK_DIM + tid;  // global thread id

    float acc = 0.0f;

    // Grid-stride over tiles; each tile processes COARSE_FACTOR items per thread.
    for (int base = gtid; base < N; base += totalThreads * COARSE_FACTOR)
    {
#pragma unroll
        for (int k = 0; k < COARSE_FACTOR; ++k)
        {
            int j = base + k * totalThreads; // <-- key fix: stride by totalThreads
            if (j < N)
                acc += in[j];
        }
    }

    // Block-wide reduction (shared memory + warp-shuffle tail)
    float sum = blockReduceSum<BLOCK_DIM>(acc, sdata);
    if (tid == 0)
        block_sums[blockIdx.x] = sum;
}

template <int BLOCK_DIM>
__global__ void finalize_reduce(const float *__restrict__ partials, int num_partials,
                                float *__restrict__ out)
{
    __shared__ float sdata[BLOCK_DIM];
    int tid = threadIdx.x;
    float x = 0.0f;

    // Grid=1: each thread may read multiple partials
    for (int i = tid; i < num_partials; i += BLOCK_DIM)
        x += partials[i];

    // Reduce to a single value
    float sum = blockReduceSum<BLOCK_DIM>(x, sdata);
    if (tid == 0)
        *out = sum;
}

// HOST
int main()
{
    const int N = 1 << 20; // 1M elements
    const int BLOCK_DIM = 256;
    const int GRID_DIM = (N + BLOCK_DIM * 2 - 1) / (BLOCK_DIM * 2);

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f; // easy to verify sum = N

    // CPU reference
    float ref = std::accumulate(h_in.begin(), h_in.end(), 0.0f);

    float *d_in, *d_partials, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_partials, GRID_DIM * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 1. Basic
    reduce_basic<BLOCK_DIM><<<GRID_DIM, BLOCK_DIM>>>(d_in, N, d_partials);
    finalize_reduce<BLOCK_DIM><<<1, BLOCK_DIM>>>(d_partials, GRID_DIM, d_out);
    float h_out1;
    cudaMemcpy(&h_out1, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 2. Contrl & Mem Optimized
    reduce_optimized<BLOCK_DIM><<<GRID_DIM, BLOCK_DIM>>>(d_in, N, d_partials);
    finalize_reduce<BLOCK_DIM><<<1, BLOCK_DIM>>>(d_partials, GRID_DIM, d_out);
    float h_out2;
    cudaMemcpy(&h_out2, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 3. Segmented + Coarsened
    reduce_segmented_coarsened<BLOCK_DIM><<<GRID_DIM, BLOCK_DIM>>>(d_in, N, d_partials);
    finalize_reduce<BLOCK_DIM><<<1, BLOCK_DIM>>>(d_partials, GRID_DIM, d_out);
    float h_out3;
    cudaMemcpy(&h_out3, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    std::cout << "Reference (CPU): " << ref << std::endl;
    std::cout << "Basic Reduction: " << h_out1 << std::endl;
    std::cout << "Optimized Reduction: " << h_out2 << std::endl;
    std::cout << "Segmented + Coarsened Reduction: " << h_out3 << std::endl;

    if (std::fabs(h_out3 - ref) < 1e-3)
        std::cout << "Results match!\n";
    else
        std::cout << "Mismatch detected!\n";

    cudaFree(d_in);
    cudaFree(d_partials);
    cudaFree(d_out);
    return 0;
}

// nvcc -O3 -arch=sm_80 -std=c++17 reduction_kernels.cu -o reduction_kernels
// ./reduction_kernels