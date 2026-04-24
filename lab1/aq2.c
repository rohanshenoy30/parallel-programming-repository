#include <stdio.h>
#include <mpi.h>

int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    int rank, size;
    int range_start, range_end;
    int total_limit = 100;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) printf("This program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 0;
    }


    range_start = (rank * 50) + 1;
    range_end = (rank + 1) * 50;

    printf("Rank %d checking range [%d, %d]:\n", rank, range_start, range_end);

    for (int i = range_start; i <= range_end; i++) {
        if (is_prime(i)) {
            printf("Rank %d found prime: %d\n", rank, i);
        }
    }

    MPI_Finalize();
    return 0;
}
