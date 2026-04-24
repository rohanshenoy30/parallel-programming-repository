#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    double a = 10.0, b = 5.0; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        if (rank == 0) printf("Error: This program requires at least 4 processes.\n");
        MPI_Finalize();
        return 0;
    }

    switch(rank) {
        case 0:
            printf("Rank 0: Addition -> %.1f + %.1f = %.1f\n", a, b, a + b);
            break;
        case 1:
            printf("Rank 1: Subtraction -> %.1f - %.1f = %.1f\n", a, b, a - b);
            break;
        case 2:
            printf("Rank 2: Multiplication -> %.1f * %.1f = %.1f\n", a, b, a * b);
            break;
        case 3:
            if (b != 0)
                printf("Rank 3: Division -> %.1f / %.1f = %.1f\n", a, b, a / b);
            else
                printf("Rank 3: Division Error -> Division by zero!\n");
            break;
        default:
            break;
    }

    MPI_Finalize();
    return 0;
}
