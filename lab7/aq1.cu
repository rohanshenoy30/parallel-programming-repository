#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void reverseWordsKernel(char* d_str, int* d_wordStarts, int* d_wordLens, int numWords) {
    int wordIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (wordIdx < numWords) {
        int start = d_wordStarts[wordIdx];
        int len = d_wordLens[wordIdx];
        
        for (int i = 0; i < len / 2; i++) {
            char temp = d_str[start + i];
            d_str[start + i] = d_str[start + len - 1 - i];
            d_str[start + len - 1 - i] = temp;
        }
    }
}

int main() {
    const char* h_input = "CUDA programming is powerful and fast";
    int n = strlen(h_input);
    char h_str[100];
    strcpy(h_str, h_input);

    int h_starts[50], h_lens[50], numWords = 0;
    
    int currentLen = 0;
    for (int i = 0; i <= n; i++) {
        if (h_str[i] == ' ' || h_str[i] == '\0') {
            if (currentLen > 0) {
                h_starts[numWords] = i - currentLen;
                h_lens[numWords] = currentLen;
                numWords++;
            }
            currentLen = 0;
        } else {
            currentLen++;
        }
    }

    char *d_str;
    int *d_starts, *d_lens;

    cudaMalloc(&d_str, n + 1);
    cudaMalloc(&d_starts, numWords * sizeof(int));
    cudaMalloc(&d_lens, numWords * sizeof(int));

    cudaMemcpy(d_str, h_str, n + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_starts, h_starts, numWords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lens, h_lens, numWords * sizeof(int), cudaMemcpyHostToDevice);

    reverseWordsKernel<<<1, numWords>>>(d_str, d_starts, d_lens, numWords);

    cudaMemcpy(h_str, d_str, n + 1, cudaMemcpyDeviceToHost);

    printf("Original: %s\n", h_input);
    printf("Reversed: %s\n", h_str);

    cudaFree(d_str); cudaFree(d_starts); cudaFree(d_lens);
    return 0;
}
