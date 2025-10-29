#include <cstdio>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b)) // ceil() helper

// y[b, j] = act( sum_i W[j,i] * x[b,i] + b[j] )
// Layouts: X [B, I], W [O, I], B [O], Y [B, O]  (row-major)
__global__ void fc_forward(const float *__restrict__ X,
                           const float *__restrict__ W,
                           const float *__restrict__ B,
                           float *__restrict__ Y,
                           int Bsz, int I, int O)
{
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x; // output neuron
    if (b >= Bsz || j >= O)
        return;
    const float *x = X + b * I;
    const float *wrow = W + j * I;

    // dot(x, wrow)
    float acc = B ? B[j] : 0.f;
    // simple unrolled loop for better ILP
    int i = 0;
    for (; i + 4 <= I; i += 4)
    {
        float4 xv = reinterpret_cast<const float4 *>(x)[i / 4];
        float4 wv = reinterpret_cast<const float4 *>(wrow)[i / 4];
        acc = fmaf(xv.x, wv.x, acc);
        acc = fmaf(xv.y, wv.y, acc);
        acc = fmaf(xv.z, wv.z, acc);
        acc = fmaf(xv.w, wv.w, acc);
    }
    for (; i < I; ++i)
        acc = fmaf(x[i], wrow[i], acc);

    // ReLU activation (swap if you want sigmoid/tanh)
    Y[b * O + j] = acc > 0.f ? acc : 0.f;
}

int main()
{
    const int B = 64, I = 120, O = 84;
    size_t xBytes = B * I * sizeof(float), wBytes = O * I * sizeof(float),
           bBytes = O * sizeof(float), yBytes = B * O * sizeof(float);

    float *x, *w, *b, *y;
    cudaMalloc(&x, xBytes);
    cudaMalloc(&w, wBytes);
    cudaMalloc(&b, bBytes);
    cudaMalloc(&y, yBytes);
    // (Fill x,w,b with your data …)

    dim3 block(256);
    dim3 grid(CEIL_DIV(O, block.x), B);
    fc_forward<<<grid, block>>>(x, w, b, y, B, I, O);
    cudaDeviceSynchronize();

    // (Read back/verify …)
    cudaFree(x);
    cudaFree(w);
    cudaFree(b);
    cudaFree(y);
    return 0;
}