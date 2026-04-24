#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#define MAX 100
__global__ void matrixMul(int *A, int *B, int *C, int n, int m, int p) {

int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < n && col < p) {
int sum = 0;
for (int k = 0; k < m; k++) {
    sum += A[row * m + k] * B[k * p + col];
}
C[row * p + col] = sum;
}
}

int main() {
int A[MAX], B[MAX], C[MAX];
int n, m, p;
printf("Enter rows and cols of Matrix A: ");
scanf("%d %d", &n, &m);
printf("Enter cols of Matrix B: ");
scanf("%d", &p);
printf("Enter Matrix A:\n");
for (int i = 0; i < n * m; i++)
    scanf("%d", &A[i]);
printf("Enter Matrix B:\n");
for (int i = 0; i < m * p; i++)
    scanf("%d", &B[i]);
int *d_A, *d_B, *d_C;
cudaMalloc(&d_A, n * m * sizeof(int));
cudaMalloc(&d_B, m * p * sizeof(int));
cudaMalloc(&d_C, n * p * sizeof(int));
cudaMemcpy(d_A, A, n * m * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, m * p * sizeof(int), cudaMemcpyHostToDevice);
dim3 block(2, 2);
dim3 grid((p + 1) / 2, (n + 1) / 2);
matrixMul<<<grid, block>>>(d_A, d_B, d_C, n, m, p);
cudaMemcpy(C, d_C, n * p * sizeof(int), cudaMemcpyDeviceToHost);
printf("Result Matrix:\n");
for (int i = 0; i < n * p; i++) {
printf("%d ", C[i]);
if ((i + 1) % p == 0) printf("\n");
}
return 0;
}