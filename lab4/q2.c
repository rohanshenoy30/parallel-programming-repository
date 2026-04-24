#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[3][3], row_buffer[3];
    int search_element, local_count = 0, total_occurrences = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) printf("Error: This program requires exactly 3 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Enter elements for the 3x3 matrix:\n");
        fflush(stdout);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("Enter the element to be searched: ");
        fflush(stdout);
        scanf("%d", &search_element);
    }

    MPI_Bcast(&search_element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    MPI_Scatter(matrix, 3, MPI_INT, row_buffer, 3, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < 3; i++) {
        if (row_buffer[i] == search_element) {
            local_count++;
        }
    }

    MPI_Reduce(&local_count, &total_occurrences, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The element %d occurred %d times in the matrix.\n", search_element, total_occurrences);
    }

    MPI_Finalize();
    return 0;
}
