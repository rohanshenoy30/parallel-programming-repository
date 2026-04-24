#include <stdio.h>
#include <cuda.h>

__global__ void oddEvenSortKernel(int *arr, int n, int phase)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    if (phase % 2 == 0)
        i = 2 * tid;
    else 
        i = 2 * tid + 1;

    if (i + 1 < n)
    {
        if (arr[i] > arr[i + 1])
        {
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

void oddEvenSortCUDA(int *h_arr, int n)
{
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n/2 + threads - 1) / threads;

    for (int phase = 0; phase < n; phase++)
    {
        oddEvenSortKernel<<<blocks, threads>>>(d_arr, n, phase);
        cudaDeviceSynchronize();  
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main()
{
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);

    oddEvenSortCUDA(arr, n);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");

    return 0;
}