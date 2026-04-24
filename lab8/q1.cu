#include <cuda_runtime.h>
#include <stdio.h>
#define ROWS 4
#define COLS 4

// kernel A
__global__ void addMatrixByRow(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col; //index calculation
            C[idx] = A[idx] + B[idx];
        }
    }
}

// kernel B
__global__ void addMatrixByCol(float *A, float *B, float *C, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        for (int row = 0; row < rows; row++) {
            int idx = row * cols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// kernel C
__global__ void addMatrixByElement(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}


void printMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int size = ROWS * COLS * sizeof(float);
    float h_A[ROWS * COLS], h_B[ROWS * COLS], h_C[ROWS * COLS];

    for (int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    addMatrixByRow<<<1, ROWS>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result (One thread per row):\n");
    printMatrix(h_C, ROWS, COLS);

    addMatrixByCol<<<1, COLS>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result (One thread per column):\n");
    printMatrix(h_C, ROWS, COLS);

    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);
    addMatrixByElement<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Result (One thread per element):\n");
    printMatrix(h_C, ROWS, COLS);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
