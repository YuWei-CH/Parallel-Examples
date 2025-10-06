#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>

// TILE_WIDTH is the side dimension of the square tile processed by each thread block.
const int TILE_WIDTH = 16;

__global__ void matrixMultiply(float *A, float *B, float *C,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns)
{
    // Calculate the row and column of the C element to work on
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of C
    if (row < numCRows && col < numCColumns)
    {
        float sum = 0.0f;

        // numAColumns == numBRows
        for (int k = 0; k < numAColumns; ++k)
        {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col] = sum;
    }
}

void readMatrix(const std::string &filename, std::vector<float> &matrix, int &rows, int &cols)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    file >> rows >> cols;
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i)
    {
        file >> matrix[i];
    }
    file.close();
}

void writeMatrix(const std::string &filename, const std::vector<float> &matrix, int rows, int cols)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        exit(1);
    }
    file << rows << " " << cols << std::endl;
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            file << matrix[r * cols + c] << (c == cols - 1 ? "" : " ");
        }
        file << std::endl;
    }
    file.close();
}

int main()
{
    // Matrix dimensions
    int numARows, numAColumns;
    int numBRows, numBColumns;

    // Allocate and initialize host matrices
    std::vector<float> hostA;
    std::vector<float> hostB;

    readMatrix("data/A.txt", hostA, numARows, numAColumns);
    readMatrix("data/B.txt", hostB, numBRows, numBColumns);

    if (numAColumns != numBRows)
    {
        std::cerr << "Matrix dimensions are incompatible for multiplication." << std::endl;
        return 1;
    }

    int numCRows = numARows;
    int numCColumns = numBColumns;

    // Allocate host C matrix
    std::vector<float> hostC(numCRows * numCColumns);

    // Allocate GPU memory
    float *A_d, *B_d, *C_d;
    unsigned int size_A = numARows * numAColumns * sizeof(float);
    unsigned int size_B = numBRows * numBColumns * sizeof(float);
    unsigned int size_C = numCRows * numCColumns * sizeof(float);

    cudaMalloc((void **)&A_d, size_A);
    cudaMalloc((void **)&B_d, size_B);
    cudaMalloc((void **)&C_d, size_C);

    // Copy memory to GPU
    cudaMemcpy(A_d, hostA.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, hostB.data(), size_B, cudaMemcpyHostToDevice);

    // Initialize Grid and Block dimensions using a 2D layout
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((numCColumns + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numCRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch GPU kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaDeviceSynchronize();

    // Copy GPU memory back to CPU
    cudaMemcpy(hostC.data(), C_d, size_C, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // Write output matrix
    writeMatrix("data/C.txt", hostC, numCRows, numCColumns);

    std::cout << "Matrix multiplication complete. Output written to data/C.txt" << std::endl;

    return 0;
}