import random
import numpy as np

def write_matrix_to_file(file_path, matrix, rows, cols):
    with open(file_path, 'w') as f:
        f.write(f"{rows} {cols}\n")
        for r in range(rows):
            f.write(" ".join(map(str, matrix[r])))
            if r < rows - 1:
                f.write("\n")

def generate_random_matrix(rows, cols):
    return np.random.rand(rows, cols) * 10

def main():
    rows = 32
    cols = 32
    
    dir_path = "/home/yuwei/Documents/UIUC-courses/CS483CUDA/matrixMultiply/data/"

    matrix_a = generate_random_matrix(rows, cols)
    matrix_b = generate_random_matrix(rows, cols)

    write_matrix_to_file(dir_path + "A.txt", matrix_a, rows, cols)
    write_matrix_to_file(dir_path + "B.txt", matrix_b, rows, cols)

    result_matrix = np.dot(matrix_a, matrix_b)

    write_matrix_to_file(dir_path + "C_answer.txt", result_matrix, rows, cols)

if __name__ == "__main__":
    main()
