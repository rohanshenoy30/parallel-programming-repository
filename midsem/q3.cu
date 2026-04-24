#include <stdio.h>
#include <cuda_runtime.h>

#define MAX_TILE 32

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

int main() {
    int N, TILE_WIDTH;

    printf("Enter N:\n");
    scanf("%d", &N);

    printf("Enter TILE_WIDTH:\n");
    scanf("%d", &TILE_WIDTH);

    int h_W[MAX_TILE];

    printf("Enter weight array:\n");
    for (int i = 0; i < TILE_WIDTH; i++) {
        scanf("%d", &h_W[i]);
    }

    // Copy weights to constant memory
    cudaMemcpyToSymbol(d_W, h_W, TILE_WIDTH * sizeof(int));

    char h_A[N * N];
    char h_B[N * N];
    char h_C[N * N];

    printf("Enter Matrix A:\n");
    for (int i = 0; i < N * N; i++) {
        scanf(" %c", &h_A[i]);
    }

    printf("Enter Matrix B:\n");
    for (int i = 0; i < N * N; i++) {
        scanf(" %c", &h_B[i]);
    }

    char *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * N * sizeof(char));
    cudaMalloc(&d_B, N * N * sizeof(char));
    cudaMalloc(&d_C, N * N * sizeof(char));

    cudaMemcpy(d_A, h_A, N * N * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(N / TILE_WIDTH, N / TILE_WIDTH);

    tiledKernel<<<grid, block>>>(d_A, d_B, d_C, N, TILE_WIDTH);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * N * sizeof(char), cudaMemcpyDeviceToHost);

    printf("\nMatrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%c ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}