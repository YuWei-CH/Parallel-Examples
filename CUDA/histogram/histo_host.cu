#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>
#include <fstream>
#include <algorithm>

// Kernels
__global__ void histo_kernel_MC(const char *buf, long size, unsigned int *histo);
__global__ void histo_kernel_SP(const char *buffer, long size, unsigned int *histo);
__global__ void histo_kernel_SP_TC2(const char *buffer, long size, unsigned int *histo);

static void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct GpuTimer
{
    cudaEvent_t startEvt{}, stopEvt{};
    GpuTimer()
    {
        cudaEventCreate(&startEvt);
        cudaEventCreate(&stopEvt);
    }
    ~GpuTimer()
    {
        cudaEventDestroy(startEvt);
        cudaEventDestroy(stopEvt);
    }
    void start() { cudaEventRecord(startEvt); }
    float stop()
    {
        cudaEventRecord(stopEvt);
        cudaEventSynchronize(stopEvt);
        float ms = 0;
        cudaEventElapsedTime(&ms, startEvt, stopEvt);
        return ms;
    }
};

static void compute_hist_cpu(const std::vector<unsigned char> &data, std::vector<unsigned int> &hist)
{
    std::fill(hist.begin(), hist.end(), 0u);
    for (unsigned char v : data)
        hist[v]++;
}

static void verify_equal(const std::vector<unsigned int> &a, const std::vector<unsigned int> &b, const char *name)
{
    if (a != b)
    {
        std::cerr << "Mismatch in " << name << std::endl;
        for (int i = 0; i < 256; ++i)
        {
            if (a[i] != b[i])
            {
                std::cerr << " bin " << i << ": " << a[i] << " vs " << b[i] << std::endl;
                break;
            }
        }
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    // Prepare input: either read from a file if a path is provided, or generate random bytes of size N.
    long N = (1L << 26); // default 64M
    std::vector<unsigned char> h_data;

    if (argc > 1)
    {
        // Try to open argv[1] as a file. If it fails, treat it as a size.
        std::ifstream fin(argv[1], std::ios::binary);
        if (fin.good())
        {
            fin.seekg(0, std::ios::end);
            std::streamoff len = fin.tellg();
            fin.seekg(0, std::ios::beg);
            if (len <= 0)
            {
                std::cerr << "Input file is empty: " << argv[1] << std::endl;
                return 1;
            }
            h_data.resize(static_cast<size_t>(len));
            fin.read(reinterpret_cast<char *>(h_data.data()), len);
            fin.close();
            N = static_cast<long>(h_data.size());
            std::cout << "Loaded input file '" << argv[1] << "' with " << N << " bytes." << std::endl;
        }
        else
        {
            N = std::atol(argv[1]);
            if (N <= 0)
            {
                std::cerr << "Invalid N or unreadable file path: '" << argv[1] << "'" << std::endl;
                return 1;
            }
        }
    }

    if (h_data.empty())
    {
        h_data.resize(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(0, 255);
        for (long i = 0; i < N; ++i)
            h_data[i] = static_cast<unsigned char>(dist(rng));
        std::cout << "Generated random input of " << N << " bytes." << std::endl;
    }

    // CPU reference
    std::vector<unsigned int> h_ref(256, 0), h_out(256, 0);
    compute_hist_cpu(h_data, h_ref);

    // Device buffers
    char *d_data = nullptr;
    unsigned int *d_hist = nullptr;
    checkCuda(cudaMalloc(&d_data, static_cast<size_t>(N)), "cudaMalloc d_data");
    checkCuda(cudaMalloc(&d_hist, 256 * sizeof(unsigned int)), "cudaMalloc d_hist");
    checkCuda(cudaMemcpy(d_data, h_data.data(), static_cast<size_t>(N), cudaMemcpyHostToDevice), "memcpy h->d data");

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x); // kernels use grid-stride loops
    grid.x = std::min(grid.x, 1024u);       // cap grid to a reasonable size

    // MC
    checkCuda(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)), "memset d_hist");
    {
        GpuTimer t;
        t.start();
        histo_kernel_MC<<<grid, block>>>(d_data, N, d_hist);
        checkCuda(cudaGetLastError(), "MC launch");
        checkCuda(cudaDeviceSynchronize(), "MC sync");
        float ms = t.stop();
        checkCuda(cudaMemcpy(h_out.data(), d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost), "memcpy d->h hist");
        verify_equal(h_out, h_ref, "MC");
        std::cout << "MC      : " << ms << " ms" << std::endl;
    }

    // SP
    checkCuda(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)), "memset d_hist");
    {
        GpuTimer t;
        t.start();
        histo_kernel_SP<<<grid, block>>>(d_data, N, d_hist);
        checkCuda(cudaGetLastError(), "SP launch");
        checkCuda(cudaDeviceSynchronize(), "SP sync");
        float ms = t.stop();
        checkCuda(cudaMemcpy(h_out.data(), d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost), "memcpy d->h hist");
        verify_equal(h_out, h_ref, "SP");
        std::cout << "SP      : " << ms << " ms" << std::endl;
    }

    // SP_TC2
    checkCuda(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)), "memset d_hist");
    {
        GpuTimer t;
        t.start();
        histo_kernel_SP_TC2<<<grid, block>>>(d_data, N, d_hist);
        checkCuda(cudaGetLastError(), "SP_TC2 launch");
        checkCuda(cudaDeviceSynchronize(), "SP_TC2 sync");
        float ms = t.stop();
        checkCuda(cudaMemcpy(h_out.data(), d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost), "memcpy d->h hist");
        verify_equal(h_out, h_ref, "SP_TC2");
        std::cout << "SP_TC2  : " << ms << " ms" << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_hist);

    std::cout << "All good." << std::endl;
    return 0;
}
