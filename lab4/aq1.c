#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    int A[5][5], B_row[5], final_B[5][5];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 5) {
        if (rank == 0) printf("Error: Run with 5 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Enter elements for a 5x5 matrix A:\n");
        fflush(stdout);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                scanf("%d", &A[i][j]);
            }
        }
    }


    MPI_Bcast(A, 25, MPI_INT, 0, MPI_COMM_WORLD);

    for (int j = 0; j < 5; j++) {
        if (rank == j) {
            B_row[j] = 0; 
        } 
        else {
            int min_val = A[0][j];
            int max_val = A[0][j];
            for (int i = 1; i < 5; i++) {
                if (A[i][j] < min_val) min_val = A[i][j];
                if (A[i][j] > max_val) max_val = A[i][j];
            }

            if (rank > j) {
                B_row[j] = max_val; 
            } else {
                B_row[j] = min_val;
            }
        }
    }

    MPI_Gather(B_row, 5, MPI_INT, final_B, 5, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nResultant Matrix B:\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                printf("%d\t", final_B[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
