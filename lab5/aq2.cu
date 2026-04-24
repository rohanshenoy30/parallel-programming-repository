#include <stdio.h>
#include <cuda_runtime.h>

__global__ void selectionSortRows(int *matrix, int rows, int cols) {
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIdx < rows) {
        int *row = &matrix[rowIdx * cols];

        for (int i = 0; i < cols - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < cols; j++) {
                if (row[j] < row[minIdx]) {
                    minIdx = j;
                }
            }
            int temp = row[minIdx];
            row[minIdx] = row[i];
            row[i] = temp;
        }
    }
}

int main() {
    int rows = 4, cols = 5;
    size_t size = rows * cols * sizeof(int);

    int h_matrix[] = {
        5, 3, 8, 1, 2,
        9, 0, 4, 7, 6,
        2, 8, 1, 5, 3,
        6, 1, 9, 2, 4
    };


    printf("Initial unsorted matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_matrix[i * cols + j]);
        }
        printf("\n");
    }

    int *d_matrix;
    cudaMalloc((void**)&d_matrix, size);
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    selectionSortRows<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols);

    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    printf("Sorted Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", h_matrix[i * cols + j]);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    return 0;
}
