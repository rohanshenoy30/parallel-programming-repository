#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int data_to_send;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("This program requires at least 2 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Master (Rank 0) starting distribution to %d slaves...\n", size - 1);

        for (int i = 1; i < size; i++) {
            data_to_send = i * 100; 
            
            printf("Master: Sending value %d to Slave %d\n", data_to_send, i);
            
            MPI_Send(&data_to_send, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        int received_number;

        MPI_Recv(&received_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        printf("Slave (Rank %d): I received the number %d from the Master.\n", rank, received_number);
    }

    MPI_Finalize();
    return 0;
}
