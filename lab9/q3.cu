#include <stdio.h>
#include <cuda_runtime.h>

__global__ void complementKernel(int *A, int *B, int M, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < M && c < N) {
        int idx = r * N + c;

        if (r > 0 && r < M - 1 && c > 0 && c < N - 1) {
            int val = A[idx];
            
            int temp = val;
            int mask = 0;
            if (val == 0) mask = 1; 
            while (temp > 0) {
                mask = (mask << 1) | 1;
                temp >>= 1;
            }
            int complement = val ^ mask;

            int binary_as_int = 0;
            int base = 1;
            while (complement > 0) {
                binary_as_int += (complement % 2) * base;
                complement /= 2;
                base *= 10;
            }
            B[idx] = binary_as_int;
        } 
        else {
            B[idx] = A[idx];
        }
    }
}

int main() {
    int M = 4, N = 4;
    size_t size = M * N * sizeof(int);

    int h_A[] = {
        1, 2, 3, 4,
        6, 5, 8, 3,
        2, 4, 10, 1,
        9, 1, 2, 5
    };
    int h_B[16];

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    complementKernel<<<grid, block>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_B[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
