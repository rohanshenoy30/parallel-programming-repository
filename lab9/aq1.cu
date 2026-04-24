#include <stdio.h>
#include <cuda_runtime.h>

__global__ void rowColSumKernel(int *A, int *B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int rowSum = 0;
        int colSum = 0;

       
        for (int k = 0; k < N; k++) {
            rowSum += A[row * N + k];     //formula stays the same just the var for row and col changes
        }

        
        for (int k = 0; k < M; k++) {
            colSum += A[k * N + col];
        }

      
        B[row * N + col] = rowSum + colSum;
    }
}

int main() {
    int M = 2; 
    int N = 3; 
    int size = M * N * sizeof(int);

    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);

    int values[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < M * N; i++) h_A[i] = values[i];

    int *d_A, *d_B;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rowColSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("Output Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_B[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
