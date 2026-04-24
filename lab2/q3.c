#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *array = NULL;
    double received_val, result;
    
    int buffer_size = size * (sizeof(double) + MPI_BSEND_OVERHEAD);
    double *buffer = (double*)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    if (rank == 0) {
        array = (double*)malloc(size * sizeof(double));
        printf("Root: Enter %d elements: ", size);
        fflush(stdout);
        for (int i = 0; i < size; i++) {
            scanf("%lf", &array[i]);
        }
        
        for (int i = 0; i < size; i++) {
            MPI_Bsend(&array[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Recv(&received_val, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank % 2 == 0) {
        result = received_val * received_val;
        printf("Process %d (Even): Square of %.2f is %.2f\n", rank, received_val, result);
    } else {
        result = cbrt(received_val); 
        printf("Process %d (Odd): Cube root of %.2f is %.2f\n", rank, received_val, result);
    }

    if (rank == 0) free(array);
    MPI_Buffer_detach(&buffer, &buffer_size);
    free(buffer);
    MPI_Finalize();
    return 0;
}
