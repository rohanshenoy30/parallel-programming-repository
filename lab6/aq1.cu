#include <stdio.h>
#include <cuda.h>

__global__ void decimalToOctal(int *d_input, int *d_output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        int num = d_input[tid];
        int octal = 0;
        int place = 1;

        while (num > 0)
        {
            int remainder = num % 8;
            octal += remainder * place;
            num = num / 8;
            place *= 10;
        }

        d_output[tid] = octal;
    }
}

int main()
{
    int n;

    printf("Enter number of elements: ");
    scanf("%d", &n);

    int h_input[n], h_output[n];

    printf("Enter %d integers:\n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", &h_input[i]);

    int *d_input, *d_output;

    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    decimalToOctal<<<blocks, threads>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nOctal values:\n");
    for (int i = 0; i < n; i++)
        printf("%d -> %d\n", h_input[i], h_output[i]);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}