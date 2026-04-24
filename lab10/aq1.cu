#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#define MAX 100
__global__ void totalCost(int *prices, int *quantities, int *total, int n) {
int i = threadIdx.x;
if (i < n) {
total[i] = prices[i] * quantities[i];
}
}
int main() {
int prices[MAX], quantities[MAX], total[MAX];
int n;
printf("Enter number of items: ");
scanf("%d", &n);
printf("Enter prices:\n");
for (int i = 0; i < n; i++)
scanf("%d", &prices[i]);
printf("Enter quantities bought:\n");
for (int i = 0; i < n; i++)
scanf("%d", &quantities[i]);
int *d_prices, *d_quantities, *d_total;
cudaMalloc(&d_prices, n * sizeof(int));
cudaMalloc(&d_quantities, n * sizeof(int));
cudaMalloc(&d_total, n * sizeof(int));
cudaMemcpy(d_prices, prices, n * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_quantities, quantities, n * sizeof(int), cudaMemcpyHostToDevice);
totalCost<<<1, n>>>(d_prices, d_quantities, d_total, n);
cudaMemcpy(total, d_total, n * sizeof(int), cudaMemcpyDeviceToHost);
int grandTotal = 0;
printf("Individual totals:\n");
for (int i = 0; i < n; i++) {
printf("%d ", total[i]);
grandTotal += total[i];
}
printf("\nGrand Total: %d\n", grandTotal);
return 0;
}