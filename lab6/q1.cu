#include <stdio.h>
#include <cuda_runtime.h>

__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, 
                                           int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < Width) {
        float Pvalue = 0;
        int N_start_point = i - (Mask_Width / 2);
        //P[i]=0.5*N[i−1]+1.0*N[i]+0.5*N[i+1]
        for (int j = 0; j < Mask_Width; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width) {  //checks if the element is a are ghost element(ignored)
                Pvalue += N[N_start_point + j] * M[j];
            }
        }
        P[i] = Pvalue;
    }
}

int main() {
    const int Width = 10;
    const int Mask_Width = 3;
    
    size_t size_N = Width * sizeof(float);
    size_t size_M = Mask_Width * sizeof(float);

    float h_N[Width] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_M[Mask_Width] = {0.5, 1.0, 0.5}; 
    float h_P[Width];

    float *d_N, *d_M, *d_P;
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_P, size_N);

    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (Width + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1D_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_M, d_P, Mask_Width, Width);

    cudaMemcpy(h_P, d_P, size_N, cudaMemcpyDeviceToHost);

    printf("Result array P: ");
    for (int i = 0; i < Width; i++) printf("%.1f ", h_P[i]);
    printf("\n");

    cudaFree(d_N); cudaFree(d_M); cudaFree(d_P);

    return 0;
}
