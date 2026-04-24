#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <device_launch_parameters.h>

#define MAX 100
__constant__ int d_mask[MAX];
__global__ void convolution(int *N, int *P, int width, int mask_width) {
int i = threadIdx.x;
int sum = 0;
int half = mask_width / 2;
if (i < width) {
for (int j = 0; j < mask_width; j++) {
    int idx = i - half + j;
    if (idx >= 0 && idx < width)
    sum += N[idx] * d_mask[j];
}
P[i] = sum;
}
}

int main() 
{
int N[MAX], M[MAX], P[MAX];
int width, mask_width;
printf("Enter input size: ");
scanf("%d", &width);
printf("Enter input array:\n");
for (int i = 0; i < width; i++)
    scanf("%d", &N[i]);
printf("Enter mask size: ");
scanf("%d", &mask_width);
printf("Enter mask array:\n");
for (int i = 0; i < mask_width; i++)
    scanf("%d", &M[i]);
int *d_N, *d_P;
cudaMalloc(&d_N, width * sizeof(int));
cudaMalloc(&d_P, width * sizeof(int));
cudaMemcpy(d_N, N, width * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpyToSymbol(d_mask, M, mask_width * sizeof(int));
convolution<<<1, width>>>(d_N, d_P, width, mask_width);
cudaMemcpy(P, d_P, width * sizeof(int), cudaMemcpyDeviceToHost);

printf("Result:\n");
for (int i = 0; i < width; i++)
    printf("%d ", P[i]);
printf("\n");
return 0;
}