#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

long long factorial(int n) {
    long long res = 1;
    for (int i = 2; i <= n; i++) {
        res *= i;
    }
    return res;
}

long long triangular_sum(int n) {
    return (long long)n * (n + 1) / 2;
}

int main(int argc, char** argv) {
    int rank, size, N;
    long long global_sum = 0;
    long long local_term_value = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the value of N (total terms): ");
        fflush(stdout); 
        if (scanf("%d", &N) != 1 || N <= 0) {
            fprintf(stderr, "Invalid N value. Exiting.\n");
            N = 0; 
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N <= 0) {
        MPI_Finalize();
        return 0;
    }

    if (size < N) {
        if (rank == 0) {
            fprintf(stderr, "Error: Need at least %d processes to cover all terms.\n", N);
        }
        MPI_Finalize();
        return 1;
    }
    
    int term_index = rank + 1;

    if (term_index <= N) {
        if (term_index % 2 != 0) {
            local_term_value = factorial(term_index);
            printf("Process %d (Term %d): Calculated factorial(%d) = %lld\n", rank, term_index, term_index, local_term_value);
        } else {
            local_term_value = triangular_sum(term_index);
            printf("Process %d (Term %d): Calculated sum to %d = %lld\n", rank, term_index, term_index, local_term_value);
        }
    }

    MPI_Reduce(&local_term_value, &global_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nAnswer:\n");
        printf("The total sum for N=%d is: **%lld**\n", N, global_sum);
    }

    MPI_Finalize();
    return 0;
}
