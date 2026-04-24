#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_WORDS 100
#define MAX_LEN 100

// Device function to check palindrome
__device__ bool isPalindrome(char *word, int len) {
    for (int i = 0; i < len / 2; i++) {
        if (word[i] != word[len - i - 1]) {
            return false;
        }
    }
    return true;
}

// Kernel
__global__ void palindromeKernel(char *words, int word_len, int num_words, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_words) {
        char *word = &words[idx * word_len];

        // find actual length (handle shorter strings)
        int len = 0;
        while (len < word_len && word[len] != '\0') len++;

        if (isPalindrome(word, len)) {
            result[idx] = 1; // mark as palindrome
        } else {
            result[idx] = 0;
        }
    }
}

int main() {
    int num_words, word_len;

    printf("Enter number of words: ");
    scanf("%d", &num_words);

    printf("Enter max length of each word: ");
    scanf("%d", &word_len);

    char h_words[MAX_WORDS][MAX_LEN];

    printf("Enter words:\n");
    for (int i = 0; i < num_words; i++) {
        scanf("%s", h_words[i]);
    }

    // Flatten 2D array
    char *h_flat = (char *)malloc(num_words * word_len * sizeof(char));
    for (int i = 0; i < num_words; i++) {
        strcpy(&h_flat[i * word_len], h_words[i]);
    }

    // Device memory
    char *d_words;
    int *d_result;

    cudaMalloc(&d_words, num_words * word_len * sizeof(char));
    cudaMalloc(&d_result, num_words * sizeof(int));

    cudaMemcpy(d_words, h_flat, num_words * word_len * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_words + blockSize - 1) / blockSize;

    palindromeKernel<<<gridSize, blockSize>>>(d_words, word_len, num_words, d_result);

    cudaDeviceSynchronize();

    // Copy result back
    int h_result[MAX_WORDS];
    cudaMemcpy(h_result, d_result, num_words * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output
    printf("\nPalindromes:\n");
    for (int i = 0; i < num_words; i++) {
        if (h_result[i] == 1) {
            printf("Index %d: %s\n", i, h_words[i]);
        }
    }

    // Free memory
    cudaFree(d_words);
    cudaFree(d_result);
    free(h_flat);

    return 0;
}