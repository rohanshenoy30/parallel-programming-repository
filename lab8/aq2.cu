#include <stdio.h>
#include <cuda_runtime.h>

#define N 4  

__device__ int getFactorial(int n) {
    if (n <= 0) return 1;
    int res = 1;
    for (int i = 2; i <= n; i++) res *= i;
    return res;
}


__device__ int getSumOfDigits(int n) {
    int sum = 0;
    n = abs(n); 
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}


__global__ void processMatrix(int *mat, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < n && j < n) {
        int idx = i * n + j;
        int val = mat[idx];

        if (i == j) {
            
            mat[idx] = 0;
        } 
        else if (i < j) {
            
            mat[idx] = getFactorial(val);
        } 
        else {
            
            mat[idx] = getSumOfDigits(val);
        }
    }
}

int main() {
    int size = N * N * sizeof(int);
    int h_A[N * N] = {
        5,  3,  4,  2,
        12, 6,  2,  5,
        25, 11, 7,  3,
        9,  18, 14, 8
    };
    int h_res[N * N];

    int *d_A;
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    processMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, N);

    cudaMemcpy(h_res, d_A, size, cudaMemcpyDeviceToHost);

    printf("Resultant Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4d ", h_res[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    return 0;
}
