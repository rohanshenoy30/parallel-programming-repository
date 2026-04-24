#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int is_non_vowel(char c) {
    if (!isalpha(c)) return 0; 
    c = tolower(c);
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
        return 0; 
    }
    return 1; 
}

int main(int argc, char** argv) {
    int rank, size;
    char *full_string = NULL;
    char *local_string = NULL;
    int n_per_process;
    int local_count = 0;
    int *all_counts = NULL;
    int total_non_vowels = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        full_string = (char*)malloc(200 * sizeof(char)); 
        printf("Enter a string (length divisible by %d): ", size);
        fflush(stdout);
        scanf("%s", full_string);

        int total_len = strlen(full_string);
        n_per_process = total_len / size;

        all_counts = (int*)malloc(size * sizeof(int));
    }

    MPI_Bcast(&n_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_string = (char*)malloc((n_per_process + 1) * sizeof(char));

    MPI_Scatter(full_string, n_per_process, MPI_CHAR, local_string, n_per_process, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n_per_process; i++) {
        if (is_non_vowel(local_string[i])) {
            local_count++;
        }
    }

    MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n--- Results ---\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d found %d non-vowels\n", i, all_counts[i]);
            total_non_vowels += all_counts[i];
        }
        printf("Total non-vowels in string: %d\n", total_non_vowels);

        free(full_string);
        free(all_counts);
    }

    free(local_string);
    MPI_Finalize();
    return 0;
}
