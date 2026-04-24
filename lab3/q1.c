#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

long long factorial(int n) {
    long long fact = 1;
    for (int i = 1; i <= n; i++) fact *= i;
    return fact;
}

int main(int argc, char** argv) {
    int rank, size;
    int number_to_process;
    long long my_factorial;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        int *input_array = (int*)malloc(size * sizeof(int));
        printf("Enter %d integers: ", size);
        fflush(stdout);
        for (int i = 0; i < size; i++) scanf("%d", &input_array[i]);

        for (int i = 0; i < size; i++) {
            if (i == 0) {
                number_to_process = input_array[0]; 
            } else {
                MPI_Send(&input_array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        free(input_array);
    } else {
        MPI_Recv(&number_to_process, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    my_factorial = factorial(number_to_process);
    printf("Process %d calculated factorial: %lld\n", rank, my_factorial);

    if (rank == 0) {
        long long total_sum = my_factorial;
        for (int i = 1; i < size; i++) {
            long long received_val;
            MPI_Recv(&received_val, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += received_val;
        }
        printf("Final Sum of Factorials: %lld\n", total_sum);
    } else {
        MPI_Send(&my_factorial, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
