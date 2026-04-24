#include <stdio.h>
#include <cuda.h>

__device__ int onesComplement(int num)
{
    int result = 0;
    int place = 1;

    while (num > 0)
    {
        int bit = num % 10;

        if (bit == 0)
            result += 1 * place;
        else
            result += 0 * place;

        num = num / 10;
        place *= 10;
    }

    return result;
}

__global__ void onesComplementKernel(int *d_input, int *d_output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
        d_output[tid] = onesComplement(d_input[tid]);
    }
}

int main()
{
    int n;

    printf("Enter number of binary elements: ");
    scanf("%d", &n);

    int h_input[n], h_output[n];

    printf("Enter %d binary numbers:\n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", &h_input[i]);

    int *d_input, *d_output;

    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    onesComplementKernel<<<blocks, threads>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nOne's Complement:\n");
    for (int i = 0; i < n; i++)
        printf("%d -> %d\n", h_input[i], h_output[i]);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}