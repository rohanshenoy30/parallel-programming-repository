#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    char input_word[100];
    char local_char;
    char *local_string;
    char *result_word = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a word of length %d: ", size);
        fflush(stdout);
        scanf("%s", input_word);
    }

    MPI_Scatter(input_word, 1, MPI_CHAR, &local_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    int count = rank + 1;
    local_string = (char *)malloc((count + 1) * sizeof(char));
    for (int i = 0; i < count; i++) {
        local_string[i] = local_char;
    }
    local_string[count] = '\0';

    int *recv_counts = NULL;
    int *displs = NULL;
    int total_length = 0;

    if (rank == 0) {
        recv_counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            recv_counts[i] = i + 1;
            displs[i] = total_length;
            total_length += recv_counts[i];
        }
        result_word = (char *)malloc((total_length + 1) * sizeof(char));
    }

    MPI_Gatherv(local_string, count, MPI_CHAR, 
                result_word, recv_counts, displs, MPI_CHAR, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        result_word[total_length] = '\0';
        printf("Output word: %s\n", result_word);
        
        free(recv_counts);
        free(displs);
        free(result_word);
    }

    free(local_string);
    MPI_Finalize();
    return 0;
}
