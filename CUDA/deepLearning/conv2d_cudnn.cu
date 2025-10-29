#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDNN_CHECK(call)                                                                                                       \
    do                                                                                                                          \
    {                                                                                                                           \
        cudnnStatus_t status = call;                                                                                            \
        if (status != CUDNN_STATUS_SUCCESS)                                                                                     \
        {                                                                                                                       \
            std::cerr << "CUDNN error at " << __FILE__ << ":" << __LINE__ << " - " << cudnnGetErrorString(status) << std::endl; \
            exit(1);                                                                                                            \
        }                                                                                                                       \
    } while (0)

#define CUDA_CHECK(call)                                                                                                     \
    do                                                                                                                       \
    {                                                                                                                        \
        cudaError_t error = call;                                                                                            \
        if (error != cudaSuccess)                                                                                            \
        {                                                                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                                                                         \
        }                                                                                                                    \
    } while (0)

float conv2d_cudnn(
    const float *h_input_data,
    const float *h_weight_data,
    float *h_output_data,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int padding,
    int stride,
    int warmup_iterations,
    int timed_iterations)
{
    // Create cuDNN handle
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Define tensor descriptors
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    // Set tensor descriptors
    int input_dims[4] = {batch_size, in_channels, in_height, in_width};
    int input_strides[4] = {
        in_channels * in_height * in_width,
        in_height * in_width,
        in_width,
        1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        input_descriptor,
        CUDNN_DATA_FLOAT,
        4,
        input_dims,
        input_strides));

    int filter_dims[4] = {out_channels, in_channels, kernel_size, kernel_size};
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
        kernel_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        4,
        filter_dims));

    int pad_a[2] = {padding, padding};
    int stride_a[2] = {stride, stride};
    int dilation_a[2] = {1, 1};
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
        convolution_descriptor,
        2,
        pad_a,
        stride_a,
        dilation_a,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // Calculate output dimensions
    int output_dims[4];
    CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
        convolution_descriptor,
        input_descriptor,
        kernel_descriptor,
        4,
        output_dims));

    int out_batch = output_dims[0];
    int out_channels_actual = output_dims[1];
    int out_height = output_dims[2];
    int out_width = output_dims[3];

    if (out_batch != batch_size || out_channels_actual != out_channels)
    {
        std::cerr << "Unexpected output dimensions from cuDNN." << std::endl;
        exit(1);
    }

    int output_strides[4] = {
        out_channels * out_height * out_width,
        out_height * out_width,
        out_width,
        1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        output_descriptor,
        CUDNN_DATA_FLOAT,
        4,
        output_dims,
        output_strides));

    // Choose convolution algorithm via heuristics
    int returned_algo_count = 0;
    const int max_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(max_algos);
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        input_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        output_descriptor,
        max_algos,
        &returned_algo_count,
        perf_results.data()));
    if (returned_algo_count == 0)
    {
        std::cerr << "cuDNN did not return any convolution algorithms." << std::endl;
        exit(1);
    }
    cudnnConvolutionFwdAlgo_t convolution_algorithm = perf_results[0].algo;

    // Get memory requirements
    size_t workspace_size = perf_results[0].memory;
    size_t queried_workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        kernel_descriptor,
        convolution_descriptor,
        output_descriptor,
        convolution_algorithm,
        &queried_workspace_size));
    if (queried_workspace_size > workspace_size)
    {
        workspace_size = queried_workspace_size;
    }

    size_t input_bytes = static_cast<size_t>(batch_size) * in_channels * in_height * in_width * sizeof(float);
    size_t weight_bytes = static_cast<size_t>(out_channels) * in_channels * kernel_size * kernel_size * sizeof(float);
    size_t output_bytes = static_cast<size_t>(batch_size) * out_channels * out_height * out_width * sizeof(float);

    // Allocate device memory
    void *d_input_data = nullptr;
    void *d_weight_data = nullptr;
    void *d_output_data = nullptr;
    void *d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_data, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight_data, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_data, output_bytes));
    if (workspace_size > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input_data, h_input_data, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_data, h_weight_data, weight_bytes, cudaMemcpyHostToDevice));

    // Perform convolution
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warm-up runs (not timed)
    for (int i = 0; i < warmup_iterations; ++i)
    {
        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn,
            &alpha,
            input_descriptor,
            d_input_data,
            kernel_descriptor,
            d_weight_data,
            convolution_descriptor,
            convolution_algorithm,
            d_workspace,
            workspace_size,
            &beta,
            output_descriptor,
            d_output_data));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Timed runs
    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));
    CUDA_CHECK(cudaEventRecord(start_event));
    for (int i = 0; i < timed_iterations; ++i)
    {
        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn,
            &alpha,
            input_descriptor,
            d_input_data,
            kernel_descriptor,
            d_weight_data,
            convolution_descriptor,
            convolution_algorithm,
            d_workspace,
            workspace_size,
            &beta,
            output_descriptor,
            d_output_data));
    }
    CUDA_CHECK(cudaEventRecord(end_event));
    CUDA_CHECK(cudaEventSynchronize(end_event));
    CUDA_CHECK(cudaGetLastError());
    float total_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, start_event, end_event));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_data, d_output_data, output_bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_input_data);
    cudaFree(d_weight_data);
    cudaFree(d_output_data);
    if (d_workspace)
    {
        cudaFree(d_workspace);
    }
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    return total_kernel_ms;
}

int main()
{
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

    // Allocate host memory
    std::vector<float> h_input(batch_size * in_channels * in_height * in_width);
    std::vector<float> h_weight(out_channels * in_channels * kernel_size * kernel_size);
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    std::vector<float> h_output(batch_size * out_channels * out_height * out_width);

    // Initialize with sample data
    for (size_t i = 0; i < h_input.size(); ++i)
    {
        h_input[i] = static_cast<float>(i % 255) / 255.0f;
    }

    for (size_t i = 0; i < h_weight.size(); ++i)
    {
        h_weight[i] = static_cast<float>(i % 10) / 10.0f;
    }

    float total_kernel_ms = conv2d_cudnn(
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
        timed_iterations);
    double avg_kernel_ms = total_kernel_ms / timed_iterations;

    double checksum = std::accumulate(h_output.begin(), h_output.end(), 0.0);

    std::cout << "Conv2D with CUDNN completed successfully!" << std::endl;
    std::cout << "Output shape: [" << batch_size << ", " << out_channels << ", "
              << out_height << ", " << out_width << "]" << std::endl;
    std::cout << std::setprecision(3) << std::fixed;
    std::cout << "Total kernel time over " << timed_iterations << " iterations: " << total_kernel_ms << " ms" << std::endl;
    std::cout << "Average kernel time per iteration: " << avg_kernel_ms << " ms" << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "Output checksum: " << checksum << std::endl;

    return 0;
}
