#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h> 

#define MAX_WORD_SIZE 100
#define TAG_WORD 1
#define TAG_TOGGLED_WORD 2

int main(int argc, char** argv) {
    int world_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        fprintf(stderr, "This program is meant to be run with exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (world_rank == 0) {
        char word[] = "HeLlO";
        char toggled_word[MAX_WORD_SIZE];
        int word_len = strlen(word) + 1; 

        printf("Process 0: Sending word: %s\n", word);
        MPI_Ssend(word, word_len, MPI_CHAR, 1, TAG_WORD, MPI_COMM_WORLD);

        MPI_Recv(toggled_word, MAX_WORD_SIZE, MPI_CHAR, 1, TAG_TOGGLED_WORD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0: Received toggled word: %s\n", toggled_word);
    } else {
        char received_word[MAX_WORD_SIZE];
        char toggled_word[MAX_WORD_SIZE];
        int i;
        MPI_Status status;

        MPI_Recv(received_word, MAX_WORD_SIZE, MPI_CHAR, 0, TAG_WORD, MPI_COMM_WORLD, &status);
        printf("Process 1: Received word: %s\n", received_word);

        for (i = 0; i < strlen(received_word); i++) {
            if (islower((unsigned char)received_word[i])) {
                toggled_word[i] = toupper((unsigned char)received_word[i]);
            } else if (isupper((unsigned char)received_word[i])) {
                toggled_word[i] = tolower((unsigned char)received_word[i]);
            } else {
                toggled_word[i] = received_word[i]; 
            }
        }
        toggled_word[i] = '\0'; 

        MPI_Ssend(toggled_word, strlen(toggled_word) + 1, MPI_CHAR, 0, TAG_TOGGLED_WORD, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
