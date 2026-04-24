#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size, N;
    int *A = NULL;
    int *sub_A = NULL;
    int local_even = 0, local_odd = 0;
    int global_even = 0, global_odd = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the total size of array N (must be divisible by %d): ", size);
        fflush(stdout);
        scanf("%d", &N);

        A = (int*)malloc(N * sizeof(int));
        printf("Enter %d elements:\n", N);
        fflush(stdout);
        for (int i = 0; i < N; i++) {
            scanf("%d", &A[i]);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = N / size;
    sub_A = (int*)malloc(chunk_size * sizeof(int));

    MPI_Scatter(A, chunk_size, MPI_INT, sub_A, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk_size; i++) {
        if (sub_A[i] % 2 == 0) {
            local_even++;
            sub_A[i] = 1; 
        } else {
            local_odd++;
            sub_A[i] = 0; 
        }
    }

    MPI_Gather(sub_A, chunk_size, MPI_INT, A, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_even, &global_even, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_odd, &global_odd, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nResultant Array A: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", A[i]);
        }
        printf("\nEven count = %d", global_even);
        printf("\nOdd count = %d\n", global_odd);

        free(A);
    }

    free(sub_A);
    MPI_Finalize();
    return 0;
}
