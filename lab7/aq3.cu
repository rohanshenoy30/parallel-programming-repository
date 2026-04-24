#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void transformKernel(char* Sin, char* T, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        char c = Sin[i];
        int numRepeats = i + 1;
        
        int offset = (i * (i + 1)) / 2;

        for (int j = 0; j < numRepeats; j++) {
            T[offset + j] = c;
        }
    }
}

int main() {
    const char* h_Sin = "Hai";
    int n = strlen(h_Sin);
    int t_len = (n * (n + 1)) / 2;

    char *d_Sin, *d_T;
    char* h_T = (char*)malloc(t_len + 1);

    cudaMalloc(&d_Sin, n);
    cudaMalloc(&d_T, t_len);

    cudaMemcpy(d_Sin, h_Sin, n, cudaMemcpyHostToDevice);

    transformKernel<<<1, n>>>(d_Sin, d_T, n);

    cudaMemcpy(h_T, d_T, t_len, cudaMemcpyDeviceToHost);
    h_T[t_len] = '\0';

    printf("Input: %s\nOutput: %s\n", h_Sin, h_T);

    cudaFree(d_Sin); cudaFree(d_T); free(h_T);
    return 0;
}
