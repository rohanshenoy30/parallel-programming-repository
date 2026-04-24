#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void repeatCharsKernel(char *d_A, int *d_B, int *d_offsets, char *d_STR, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalElements) {
        char c = d_A[idx];
        int count = d_B[idx];
        int startPos = d_offsets[idx];

        for (int i = 0; i < count; i++) {
            d_STR[startPos + i] = c;
        }
    }
}

int main() {
    int M = 2, N = 4;
    int totalElements = M * N;

    char h_A[] = {'p', 'C', 'a', 'P', 'e', 'X', 'a', 'M'};
    int h_B[] = {1, 2, 4, 3, 2, 4, 3, 2};
    
    int *h_offsets = (int*)malloc(totalElements * sizeof(int));
    int currentOffset = 0;
    for (int i = 0; i < totalElements; i++) {
        h_offsets[i] = currentOffset;
        currentOffset += h_B[i];
    }
    int totalStringLength = currentOffset;

    char *d_A, *d_STR;
    int *d_B, *d_offsets;

    cudaMalloc((void**)&d_A, totalElements * sizeof(char));
    cudaMalloc((void**)&d_B, totalElements * sizeof(int));
    cudaMalloc((void**)&d_offsets, totalElements * sizeof(int));
    cudaMalloc((void**)&d_STR, (totalStringLength + 1) * sizeof(char));

    cudaMemcpy(d_A, h_A, totalElements * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, totalElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, totalElements * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    
    repeatCharsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_offsets, d_STR, totalElements);

    char *h_STR = (char*)malloc((totalStringLength + 1) * sizeof(char));
    cudaMemcpy(h_STR, d_STR, totalStringLength * sizeof(char), cudaMemcpyDeviceToHost);
    h_STR[totalStringLength] = '\0';

    printf("Output String STR: %s\n", h_STR);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_offsets); cudaFree(d_STR);
    free(h_offsets); free(h_STR);

    return 0;
}
