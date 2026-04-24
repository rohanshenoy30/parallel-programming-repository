#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    int rank, size;
    char *s1 = NULL, *s2 = NULL, *result = NULL;
    char *local_s1, *local_s2, *local_res;
    int total_len, chunk_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        s1 = (char*)malloc(200 * sizeof(char));
        s2 = (char*)malloc(200 * sizeof(char));

        printf("Enter string S1: ");
        fflush(stdout);
        scanf("%s", s1);
        printf("Enter string S2: ");
        fflush(stdout);
        scanf("%s", s2);

        total_len = strlen(s1);
        chunk_size = total_len / size;
        
        result = (char*)malloc((2 * total_len + 1) * sizeof(char));
    }

    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_s1 = (char*)malloc(chunk_size * sizeof(char));
    local_s2 = (char*)malloc(chunk_size * sizeof(char));
    local_res = (char*)malloc((2 * chunk_size) * sizeof(char));

    MPI_Scatter(s1, chunk_size, MPI_CHAR, local_s1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, chunk_size, MPI_CHAR, local_s2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);


    for (int i = 0; i < chunk_size; i++) {
        local_res[2 * i] = local_s1[i];     
        local_res[2 * i + 1] = local_s2[i]; 
    }

    MPI_Gather(local_res, 2 * chunk_size, MPI_CHAR, result, 2 * chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        result[2 * total_len] = '\0'; 
        printf("\nString S1: %s", s1);
        printf("\nString S2: %s", s2);
        printf("\nResultant string: %s\n", result);

        free(s1);
        free(s2);
        free(result);
    }

    free(local_s1);
    free(local_s2);
    free(local_res);

    MPI_Finalize();
    return 0;
}
