#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#define MAX 100
__global__ void tiledConv(int *N, int *M, int *P, int width, int mask_width) {
__shared__ int tile[MAX];       //tile is a shared array
int tx = threadIdx.x;
int i = blockIdx.x * blockDim.x + tx;
int half = mask_width / 2;
tile[tx] = (i < width) ? N[i] : 0;   //tile value is assigned as matrix value or 0 if out of bounds
__syncthreads();
int sum = 0;
if (i < width) {                                //this part is same as the normal convolution
for (int j = 0; j < mask_width; j++) {
    int idx = tx - half + j;
    if (idx >= 0 && idx < blockDim.x)
    sum += tile[idx] * M[j];    //convolution with mask and tile
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
int *d_N, *d_M, *d_P;
cudaMalloc(&d_N, width * sizeof(int));
cudaMalloc(&d_M, mask_width * sizeof(int));
cudaMalloc(&d_P, width * sizeof(int));
cudaMemcpy(d_N, N, width * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_M, M, mask_width * sizeof(int), cudaMemcpyHostToDevice);
tiledConv<<<1, width>>>(d_N, d_M, d_P, width, mask_width);
cudaMemcpy(P, d_P, width * sizeof(int), cudaMemcpyDeviceToHost);

printf("Result:\n");
for (int i = 0; i < width; i++)
    printf("%d ", P[i]);
printf("\n");
return 0;
}