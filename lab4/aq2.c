#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, n, local_n;
    double pi_val, local_pi, h, sum, x;
    double local_a, local_b, a = 0.0, b = 1.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the number of intervals (n): ");
        fflush(stdout);
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = (b - a) / n;
    
    local_n = n / size;
    local_a = a + rank * local_n * h;
    local_b = local_a + local_n * h;

    sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        x = local_a + (i + 0.5) * h;
        sum += 4.0 / (1.0 + x * x);
    }
    local_pi = h * sum;

    MPI_Reduce(&local_pi, &pi_val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("With n = %d intervals, pi is approximately %.16f\n", n, pi_val);
    }

    MPI_Finalize();
    return 0;
}
