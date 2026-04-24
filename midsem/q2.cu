#include <stdio.h>
#include <cuda_runtime.h>

#define MAX 100

// Constant memory for dimensions
__constant__ int d_M;
__constant__ int d_N;

// Device function for repetition count
__device__ int getRepeat(int val) {
    return (val % 2 == 0) ? 2 : 1;
}

// Kernel
__global__ void buildString(char *A, int *B, int *prefix, char *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d_M * d_N;

    if (idx < total) {
        char ch = A[idx];
        int repeat = getRepeat(B[idx]);

        int start = prefix[idx];

        // Write repeated characters
        for (int i = 0; i < repeat; i++) {
            output[start + i] = ch;
        }
    }
}

int main() {
    int M = 2, N = 3;
    int total = M * N;

    // Host arrays
    char h_A[] = {'a','b','c','d','e','f'};
    int  h_B[] = {1,2,3,4,5,6};

    // Step 1: Compute repeat counts
    int repeat[MAX];
    for (int i = 0; i < total; i++) {
        repeat[i] = (h_B[i] % 2 == 0) ? 2 : 1;
    }

    // Step 2: Prefix sum
    int prefix[MAX];
    prefix[0] = 0;
    for (int i = 1; i < total; i++) {
        prefix[i] = prefix[i-1] + repeat[i-1];
    }

    int output_size = prefix[total-1] + repeat[total-1];

    // Allocate output
    char h_output[MAX];

    // Device memory
    char *d_A, *d_output;
    int *d_B, *d_prefix;

    cudaMalloc(&d_A, total * sizeof(char));
    cudaMalloc(&d_B, total * sizeof(int));
    cudaMalloc(&d_prefix, total * sizeof(int));
    cudaMalloc(&d_output, output_size * sizeof(char));

    cudaMemcpy(d_A, h_A, total * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix, total * sizeof(int), cudaMemcpyHostToDevice);

    // Copy M, N to constant memory
    cudaMemcpyToSymbol(d_M, &M, sizeof(int));
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    buildString<<<gridSize, blockSize>>>(d_A, d_B, d_prefix, d_output);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output, d_output, output_size * sizeof(char), cudaMemcpyDeviceToHost);

    // Print output string
    printf("Output string:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%c", h_output[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_prefix);
    cudaFree(d_output);

    return 0;
}