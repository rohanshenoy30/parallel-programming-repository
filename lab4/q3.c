#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[4][4], row[4], prev_row[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) printf("Error: This program requires exactly 4 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Enter elements for a 4x4 matrix:\n");
        fflush(stdout);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }

    MPI_Scatter(matrix, 4, MPI_INT, row, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        MPI_Send(row, 4, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } 
    else {
        MPI_Recv(prev_row, 4, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for (int i = 0; i < 4; i++) {
            row[i] += prev_row[i];
        }

        if (rank < 3) {
            MPI_Send(row, 4, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Gather(row, 4, MPI_INT, matrix, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nOutput Matrix:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
