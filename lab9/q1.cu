#include <stdio.h>
#include <cuda_runtime.h>

__global__ void spmv_csr_kernel(int num_rows, const float* data, const int* col_index, const int* row_ptr, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float dot_product = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int i = row_start; i < row_end; i++) {
            dot_product += data[i] * x[col_index[i]];
        }
        y[row] = dot_product;
    }
}

int main() {
    const int num_rows = 3;
    const int num_cols = 3;
    const int nnz = 6; 

    float h_data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    int h_col_index[] = {0, 2, 0, 1, 2, 2}; 
    int h_row_ptr[] = {0, 2, 5, 6};         

    float h_x[] = {1.0f, 2.0f, 3.0f};
    float h_y[3] = {0.0f}; 

    
    float *d_data, *d_x, *d_y;
    int *d_col_index, *d_row_ptr;

    cudaMalloc((void**)&d_data, nnz * sizeof(float));
    cudaMalloc((void**)&d_col_index, nnz * sizeof(int));
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));

   
    cudaMemcpy(d_data, h_data, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, h_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);

 
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    
    spmv_csr_kernel<<<gridSize, blockSize>>>(num_rows, d_data, d_col_index, d_row_ptr, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result Vector Y:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("Y[%d] = %.2f\n", i, h_y[i]);
    }

    cudaFree(d_data); 
    cudaFree(d_col_index); 
    cudaFree(d_row_ptr);
    cudaFree(d_x); 
    cudaFree(d_y);

    return 0;
}
