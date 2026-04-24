#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    int M;
    float *full_array = NULL;
    float *local_array = NULL;
    float local_sum = 0.0, local_avg;
    float *all_averages = NULL;
    float total_avg_sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the number of elements per process (M): ");
        fflush(stdout);
        scanf("%d", &M);
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_array = (float*)malloc(M * sizeof(float));

    if (rank == 0) {
        int total_elements = size * M;
        full_array = (float*)malloc(total_elements * sizeof(float));
        
        printf("Enter %d elements:\n", total_elements);
        for (int i = 0; i < total_elements; i++) {
            scanf("%f", &full_array[i]);
        }
        
        all_averages = (float*)malloc(size * sizeof(float));
    }

    MPI_Scatter(full_array, M, MPI_FLOAT, local_array, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < M; i++) {
        local_sum += local_array[i];
    }
    local_avg = local_sum / M;
    printf("Process %d: Local Average = %.2f\n", rank, local_avg);

    MPI_Gather(&local_avg, 1, MPI_FLOAT, all_averages, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            total_avg_sum += all_averages[i];
        }
        float final_avg = total_avg_sum / size;
        printf("\n>>> Total Average of all elements: %.2f <<<\n", final_avg);

        free(full_array);
        free(all_averages);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
