#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void countWordKernel(const char* sentence, int sentenceLen, const char* target, int targetLen, int* totalCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx <= (sentenceLen - targetLen)) {
        bool match = true;

        for (int i = 0; i < targetLen; i++) {
            if (sentence[idx + i] != target[i]) {
                match = false;
                break;
            }
        }

        if (match) {
            bool prefixOk = (idx == 0) || (sentence[idx - 1] == ' ');
            bool suffixOk = (idx + targetLen == sentenceLen) || (sentence[idx + targetLen] == ' ');

            if (prefixOk && suffixOk) {
                atomicAdd(totalCount, 1);
            }
        }
    }
}

int main() {
    const char* h_sentence = "the quick brown fox jumps over the lazy dog the";
    const char* h_target = "the";

    int sentenceLen = strlen(h_sentence);
    int targetLen = strlen(h_target);

 
    char *d_sentence, *d_target;
    int *d_count;
    int h_count = 0;

    cudaMalloc((void**)&d_sentence, sentenceLen + 1);
    cudaMalloc((void**)&d_target, targetLen + 1);
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_sentence, h_sentence, sentenceLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, targetLen + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (sentenceLen + threadsPerBlock - 1) / threadsPerBlock;

    countWordKernel<<<blocksPerGrid, threadsPerBlock>>>(d_sentence, sentenceLen, d_target, targetLen, d_count);

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appeared %d times.\n", h_target, h_count);

    cudaFree(d_sentence);
    cudaFree(d_target);
    cudaFree(d_count);

    return 0;
}
