#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void transformStringKernel(char* S, char* RS, int N) {
    int i = threadIdx.x; 
    if (i < N) {
        int offset = 0;
        for (int j = 0; j < i; j++) {
            offset += (N - j);    //calculate the offset for the current thread 

        }

        int charsToCopy = N - i;        //each thread writes N-i characters to the output string
        for (int j = 0; j < charsToCopy; j++) {
            RS[offset + j] = S[j];
        }
    }
}

int main() {
    char h_S[] = "PCAP";
    int N = strlen(h_S);
    int RS_len = (N * (N + 1)) / 2;

    char *d_S, *d_RS;
    char h_RS[RS_len + 1];

    cudaMalloc((void**)&d_S, N * sizeof(char));
    cudaMalloc((void**)&d_RS, RS_len * sizeof(char));

    cudaMemcpy(d_S, h_S, N * sizeof(char), cudaMemcpyHostToDevice);

    transformStringKernel<<<1, N>>>(d_S, d_RS, N);

    cudaMemcpy(h_RS, d_RS, RS_len * sizeof(char), cudaMemcpyDeviceToHost);
    h_RS[RS_len] = '\0'; 

    printf("Input string S: %s\n", h_S);
    printf("Output string RS: %s\n", h_RS);

    cudaFree(d_S);
    cudaFree(d_RS);
    return 0;
}
