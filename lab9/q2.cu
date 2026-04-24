#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void transformMatrixKernel(float* matrix, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int index = row * N + col;
        
        float power = (float)(row + 1);
        matrix[index] = powf(matrix[index], power);
    }
}

int main() {
    int M = 4; 
    int N = 3; 
    size_t size = M * N * sizeof(float);

    
    float* h_A = (float*)malloc(size);

    printf("Original Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (float)(j + 2); 
            printf("%.1f\t", h_A[i * N + j]);
        }
        printf("\n");
    }

    float* d_A;
    cudaMalloc((void**)&d_A, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transformMatrixKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, M, N);
    
    cudaDeviceSynchronize();

    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    printf("\nTransformed Matrix (Row 1: same, Row 2: sq, Row 3: cube...):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f\t", h_A[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    free(h_A);

    return 0;
}
