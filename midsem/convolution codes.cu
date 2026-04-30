//normal convolution

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
