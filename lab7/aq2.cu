#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void concatKernel(char* d_Sin, char* d_Sout, int len, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < len) {
        char c = d_Sin[i];
        for (int j = 0; j < N; j++) {
            int target_idx = (j * len) + i;
            d_Sout[target_idx] = c;
        }
    }
}

int main() {
    const char* h_Sin = "Hello";
    int N = 3;
    int len = strlen(h_Sin);
    int total_len = len * N;

    char *d_Sin, *d_Sout;
    char *h_Sout = (char*)malloc(total_len + 1);

    cudaMalloc((void**)&d_Sin, len);
    cudaMalloc((void**)&d_Sout, total_len);

    cudaMemcpy(d_Sin, h_Sin, len, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    concatKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Sin, d_Sout, len, N);

    cudaMemcpy(h_Sout, d_Sout, total_len, cudaMemcpyDeviceToHost);
    h_Sout[total_len] = '\0'; 

    printf("Input: Sin = \"%s\"  N = %d\n", h_Sin, N);
    printf("Output: Sout = \"%s\"\n", h_Sout);

    cudaFree(d_Sin);
    cudaFree(d_Sout);
    free(h_Sout);

    return 0;
}
