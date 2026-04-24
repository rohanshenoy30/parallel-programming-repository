#include <stdio.h>
#include <mpi.h>

int reverse_integer(int n) {
    int reversed = 0;
    while (n != 0) {
        reversed = reversed * 10 + n % 10;
        n /= 10;
    }
    return reversed;
}

int main(int argc, char** argv) {
    int rank, size;
    int input_array[9] = {18, 523, 301, 1234, 2, 14, 108, 150, 1928};
    int output_array[9];
    int local_val;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 9) {
        if (rank == 0) printf("Error: This program requires exactly 9 processes.\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Scatter(input_array, 1, MPI_INT, &local_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_reversed = reverse_integer(local_val);

    MPI_Gather(&local_reversed, 1, MPI_INT, output_array, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Output: ");
        for (int i = 0; i < 9; i++) {
            printf("%d%s", output_array[i], (i == 8) ? "" : ", ");
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
