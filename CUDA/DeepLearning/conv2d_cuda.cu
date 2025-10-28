#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// CUDA kernel for 2D convolution
__global__ void conv2d_kernel(
    const float* input, 
    const float* weight, 
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int padding,
    int stride
) {
    // Calculate output position
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int block_z = blockIdx.z;
    int batch = block_z / out_channels;
    int out_c = block_z % out_channels;
    
    if (out_x >= out_width || out_y >= out_height || out_c >= out_channels || batch >= batch_size) 
        return;
    
    float sum = 0.0f;
    
    // Convolve kernel with input
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate input position
                int in_x = out_x * stride - padding + kx;
                int in_y = out_y * stride - padding + ky;
                
                // Check boundary conditions
                if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                    // Get input value
                    int input_idx = ((batch * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                    // Get weight value
                    int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Write output
    int output_idx = ((batch * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    output[output_idx] = sum;
}

// Host function to launch convolution kernel
float conv2d_cuda(
    const float* h_input,
    const float* h_weight,
    float* h_output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int padding,
    int stride,
    int warmup_iterations,
    int timed_iterations
) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    size_t input_size = batch_size * in_channels * in_height * in_width * sizeof(float);
    size_t weight_size = out_channels * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = batch_size * out_channels * out_height * out_width * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, weight_size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 block_size(16, 16, 1);
    dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        batch_size * out_channels
    );
    
    // Warm-up runs (not timed)
    for (int i = 0; i < warmup_iterations; ++i) {
        conv2d_kernel<<<grid_size, block_size>>>(
            d_input, d_weight, d_output,
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            kernel_size, padding, stride
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Timed runs
    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));
    CUDA_CHECK(cudaEventRecord(start_event));
    for (int i = 0; i < timed_iterations; ++i) {
        conv2d_kernel<<<grid_size, block_size>>>(
            d_input, d_weight, d_output,
            batch_size, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            kernel_size, padding, stride
        );
    }
    CUDA_CHECK(cudaEventRecord(end_event));
    CUDA_CHECK(cudaEventSynchronize(end_event));
    CUDA_CHECK(cudaGetLastError());
    float total_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, start_event, end_event));
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));

    return total_kernel_ms;
}

// Example usage
int main() {
    // Example parameters
    const int batch_size = 8;
    const int in_channels = 64;
    const int in_height = 512;
    const int in_width = 512;
    const int out_channels = 128;
    const int kernel_size = 3;
    const int padding = 1;
    const int stride = 1;
    const int warmup_iterations = 5;
    const int timed_iterations = 50;
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Allocate host memory
    std::vector<float> h_input(batch_size * in_channels * in_height * in_width);
    std::vector<float> h_weight(out_channels * in_channels * kernel_size * kernel_size);
    std::vector<float> h_output(batch_size * out_channels * out_height * out_width);
    
    // Initialize with sample data
    for (size_t i = 0; i < h_input.size(); ++i) {
        h_input[i] = static_cast<float>(i % 255) / 255.0f;
    }
    
    for (size_t i = 0; i < h_weight.size(); ++i) {
        h_weight[i] = static_cast<float>(i % 10) / 10.0f;
    }
    
    float total_kernel_ms = conv2d_cuda(
        h_input.data(), 
        h_weight.data(), 
        h_output.data(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        padding,
        stride,
        warmup_iterations,
        timed_iterations
    );
    double avg_kernel_ms = total_kernel_ms / timed_iterations;
    
    double checksum = std::accumulate(h_output.begin(), h_output.end(), 0.0);
    
    std::cout << "Conv2D with pure CUDA completed successfully!" << std::endl;
    std::cout << "Output shape: [" << batch_size << ", " << out_channels << ", " 
              << out_height << ", " << out_width << "]" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total kernel time over " << timed_iterations << " iterations: "
              << total_kernel_ms << " ms" << std::endl;
    std::cout << "Average kernel time per iteration: "
              << avg_kernel_ms << " ms" << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "Output checksum: " << checksum << std::endl;
    
    return 0;
}
