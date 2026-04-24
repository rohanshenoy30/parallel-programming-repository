#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int send_data = 101;
    int recv_data = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("This program must be run with exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    printf("Process %d attempting synchronous send and receive...\n", rank);

    MPI_Ssend(&send_data, 1, MPI_INT, 1 - rank, 0, MPI_COMM_WORLD); 
    printf("Process %d finished sending (will not print if deadlocked here).\n", rank);

    MPI_Recv(&recv_data, 1, MPI_INT, 1 - rank, 0, MPI_COMM_WORLD, &status);
    printf("Process %d received data: %d\n", rank, recv_data);

    MPI_Finalize();
    return 0;
}
