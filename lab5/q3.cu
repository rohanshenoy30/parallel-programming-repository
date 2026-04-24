#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void computeSine(float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = sinf(input[i]);
    }
}

int main() {
    int n = 10; 
    size_t size = n * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_in[i] = i * (M_PI / (n - 1));
    }

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    computeSine<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("Radians \t Sine\n");
    for (int i = 0; i < n; i++) {
        printf("%.4f \t %.4f\n", h_in[i], h_out[i]);
    }

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}
