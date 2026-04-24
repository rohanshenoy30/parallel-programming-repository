#include <stdio.h>
#include <cuda_runtime.h>

#define M 4
#define K 3
#define N 4

// kernel A
__global__ void matMulByRow(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += A[row * k + l] * B[l * n + j];
            }
            C[row * n + j] = sum;
        }
    }
}

// kernel B
__global__ void matMulByCol(float *A, float *B, float *C, int m, int k, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        for (int i = 0; i < m; i++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + col];
            }
            C[i * n + col] = sum;
        }
    }
}

// kernel C
__global__ void matMulByElement(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

void printMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) printf("%.1f ", mat[i * cols + j]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float h_A[M * K], h_B[K * N], h_C[M * N];
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f; // Matrix A of 1s
    for (int i = 0; i < K * N; i++) h_B[i] = 2.0f; // Matrix B of 2s

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA); cudaMalloc(&d_B, sizeB); cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    matMulByRow<<<1, M>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("Method A (One thread per row):\n");
    printMatrix(h_C, M, N);

    matMulByCol<<<1, N>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("Method B (One thread per column):\n");
    printMatrix(h_C, M, N);

    dim3 threads(2, 2);
    dim3 blocks((N + 1) / 2, (M + 1) / 2);
    matMulByElement<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("Method C (One thread per element):\n");
    printMatrix(h_C, M, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
