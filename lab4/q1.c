#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

long long factorial(int n) {
    long long fact = 1;
    for (int i = 1; i <= n; i++) fact *= i;
    return fact;
}

int main(int argc, char** argv) {
    int rank, size, err_code;
    long long local_fact, prefix_sum;

    err_code = MPI_Init(&argc, &argv);
    if (err_code != MPI_SUCCESS) {
        fprintf(stderr, "Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, err_code);
    }

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    err_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (err_code != MPI_SUCCESS) goto handle_error;

    err_code = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (err_code != MPI_SUCCESS) goto handle_error;


    local_fact = factorial(rank + 1);

    err_code = MPI_Scan(&local_fact, &prefix_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    if (err_code != MPI_SUCCESS) {
        handle_error: {
            char err_buffer[MPI_MAX_ERROR_STRING];
            int resultlen;
            MPI_Error_string(err_code, err_buffer, &resultlen);
            fprintf(stderr, "Rank %d encountered MPI Error: %s\n", rank, err_buffer);
            MPI_Abort(MPI_COMM_WORLD, err_code);
        }
    }

    if (rank == size - 1) {
        printf("The total sum of factorials from 1! to %d! is: %lld\n", size, prefix_sum);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
