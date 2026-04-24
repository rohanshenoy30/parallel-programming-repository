#include <stdio.h>
#include <cuda.h>
#include <limits.h>

__global__ void findMinKernel(int *arr, int n, int start, int *minVal, int *minIdx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int index = start + tid;

    if (index < n)
    {
        int value = arr[index];

        int old = atomicMin(minVal, value);

        if (value < old)
        {
            atomicExch(minIdx, index);
        }
    }
}

void selectionSortCUDA(int *h_arr, int n)
{
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int *d_minVal, *d_minIdx;
    cudaMalloc((void**)&d_minVal, sizeof(int));
    cudaMalloc((void**)&d_minIdx, sizeof(int));

    for (int i = 0; i < n - 1; i++)
    {
        int minVal = INT_MAX;
        int minIdx = i;

        cudaMemcpy(d_minVal, &minVal, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_minIdx, &minIdx, sizeof(int), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (n - i + threads - 1) / threads;

        findMinKernel<<<blocks, threads>>>(d_arr, n, i, d_minVal, d_minIdx);
        cudaDeviceSynchronize();

        cudaMemcpy(&minIdx, d_minIdx, sizeof(int), cudaMemcpyDeviceToHost);

        int temp1, temp2;
        cudaMemcpy(&temp1, &d_arr[i], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&temp2, &d_arr[minIdx], sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(&d_arr[i], &temp2, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_arr[minIdx], &temp1, sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_minVal);
    cudaFree(d_minIdx);
}

int main()
{
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);

    selectionSortCUDA(arr, n);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");

    return 0;
}