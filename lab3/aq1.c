#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size, M;
    int *full_array = NULL;
    int *sub_array = NULL;
    int *result_array = NULL;
    int *final_results = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the value of M (elements per process): ");
        fflush(stdout);
        scanf("%d", &M);

        full_array = (int*)malloc(size * M * sizeof(int));
        final_results = (int*)malloc(size * M * sizeof(int));

        printf("Enter %d elements:\n", size * M);
        fflush(stdout);
        for (int i = 0; i < size * M; i++) {
            scanf("%d", &full_array[i]);
        }
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    sub_array = (int*)malloc(M * sizeof(int));
    result_array = (int*)malloc(M * sizeof(int));

    MPI_Scatter(full_array, M, MPI_INT, sub_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    int power = rank + 2;
    for (int i = 0; i < M; i++) {
        result_array[i] = (int)pow(sub_array[i], power);
    }

    MPI_Gather(result_array, M, MPI_INT, final_results, M, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nProcessed results (Rank 0: squared, Rank 1: cubed, etc.):\n");
        for (int i = 0; i < size * M; i++) {
            printf("%d ", final_results[i]);
            if ((i + 1) % M == 0) printf("| "); 
        }
        printf("\n");

        free(full_array);
        free(final_results);
    }

    free(sub_array);
    free(result_array);
    MPI_Finalize();
    return 0;
}
