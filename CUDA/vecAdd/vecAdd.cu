#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

__global__ void vecAdd(float *in1, float *in2, float *out, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
        out[i] = in1[i] + in2[i];
}

void readData(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    float num;
    while (file >> num) {
        data.push_back(num);
    }
    file.close();
}

void writeData(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        exit(1);
    }
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i] << (i == data.size() - 1 ? "" : " ");
    }
    file.close();
}

int main() {
    std::vector<float> hostInput1;
    std::vector<float> hostInput2;

    readData("data/input1.txt", hostInput1);
    readData("data/input2.txt", hostInput2);

    if (hostInput1.size() != hostInput2.size()) {
        std::cerr << "Input vectors must have the same size." << std::endl;
        return 1;
    }

    int inputLength = hostInput1.size();
    std::vector<float> hostOutput(inputLength);

    // Allocate GPU memory
    unsigned int size = inputLength * sizeof(float);
    float *in1_d, *in2_d, *out_d;
    cudaMalloc((void **)&in1_d, size);
    cudaMalloc((void **)&in2_d, size);
    cudaMalloc((void **)&out_d, size);

    // Copy memory to the GPU
    cudaMemcpy(in1_d, hostInput1.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(in2_d, hostInput2.data(), size, cudaMemcpyHostToDevice);

    // Initialize the grid and block dim
    const unsigned int numberThreadPerBlock = 512;
    const unsigned int numBlocks = (inputLength + numberThreadPerBlock - 1) / numberThreadPerBlock;
    dim3 DimGrid(numBlocks, 1, 1);
    dim3 DimBlock(numberThreadPerBlock, 1, 1);

    // Launch GPU kernel
    vecAdd<<<DimGrid, DimBlock>>>(in1_d, in2_d, out_d, inputLength);

    cudaDeviceSynchronize();

    // Copy GPU memory back to CPU
    cudaMemcpy(hostOutput.data(), out_d, size, cudaMemcpyDeviceToHost);

    // Write output data
    writeData("data/output.txt", hostOutput);

    // Free GPU memory
    cudaFree(in1_d);
    cudaFree(in2_d);
    cudaFree(out_d);

    std::cout << "Vector addition complete. Output written to data/output.txt" << std::endl;

    return 0;
}
