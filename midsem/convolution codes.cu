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





__constant__ int d_W[MAX_TILE]; // constant memory for weights

__global__ void tiledKernel(char *A, char *B, char *C, int N, int TILE_WIDTH) {

    __shared__ char sA[MAX_TILE][MAX_TILE];
    __shared__ char sB[MAX_TILE][MAX_TILE];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int sum = 0;

    // Loop over tiles
    for (int t = 0; t < N / TILE_WIDTH; t++) {

        // Load tiles into shared memory
        sA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_WIDTH + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE_WIDTH; k++) {

            int a_val = (int)sA[threadIdx.y][k];
            int b_val = (int)sB[k][threadIdx.x];

            int weight = d_W[(t * TILE_WIDTH + k) % TILE_WIDTH];

            sum += (a_val + b_val) * weight;
        }

        __syncthreads();
    }

    // Final mod + convert to char
    sum = sum % 26;
    C[row * N + col] = (char)(sum + 'A');
}
