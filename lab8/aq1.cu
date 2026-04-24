#include <stdio.h>
#include <cuda_runtime.h>

#define M 2
#define N 3

__global__ void transformMatrix(int *A, int *B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int val = A[row * n + col];
        int sum = 0;

        if (val % 2 == 0) {
            for (int j = 0; j < n; j++) {
                sum += A[row * n + j];
            }
        } else {
            for (int i = 0; i < m; i++) {
                sum += A[i * n + col];
            }
        }
        B[row * n + col] = sum;
    }
}

int main() {
    int size = M * N * sizeof(int);
    int h_A[M * N] = {1, 2, 3, 4, 5, 6}; 
    int h_B[M * N];

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transformMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("Input Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) printf("%d ", h_A[i * N + j]);
        printf("\n");
    }

    printf("\nResultant Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) printf("%d ", h_B[i * N + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
