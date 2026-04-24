#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Function to check if a number is prime
bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i = i + 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int rank, size;
    int N; 
    int *global_array = NULL;
    int *global_results = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the number of elements (N): ");
        fflush(stdout);
        if (scanf("%d", &N) != 1 || N <= 0) {
            printf("Invalid N. Using default N=10.\n");
            N = 10;
        }
        
        global_array = (int*)malloc(N * sizeof(int));
        global_results = (int*)malloc(N * sizeof(int)); 
        
        printf("Enter %d elements:\n", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &global_array[i]);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N % size != 0) {
        if (rank == 0) {
            printf("Warning: N is not divisible by the number of processes. This simple scatter might not work as expected.\n");
        }
        MPI_Finalize();
        return 0;
    }
    int elements_per_proc = N / size;

    int *local_array = (int*)malloc(elements_per_proc * sizeof(int));
    int *local_results = (int*)malloc(elements_per_proc * sizeof(int));

    MPI_Scatter(global_array, elements_per_proc, MPI_INT,
                local_array, elements_per_proc, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < elements_per_proc; i++) {
        if (is_prime(local_array[i])) {
            local_results[i] = 1; // Mark as prime
        } else {
            local_results[i] = 0; // Mark as not prime
        }
    }

    MPI_Gather(local_results, elements_per_proc, MPI_INT,
               global_results, elements_per_proc, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nPrimality check results:\n");
        for (int i = 0; i < N; i++) {
            printf("Element %d (%d) is %s\n", i, global_array[i], 
                   (global_results[i] ? "prime" : "not prime"));
        }
        free(global_array);
        free(global_results);
    }

    free(local_array);
    free(local_results);

    MPI_Finalize();
    return 0;
}
