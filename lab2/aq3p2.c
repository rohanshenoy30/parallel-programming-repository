#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MESSAGE_SIZE 1000000 

int main(int argc, char **argv) {
    int rank, size;
  
    int *send_buffer, *recv_buffer;
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
    
    send_buffer = (int*)malloc(MESSAGE_SIZE * sizeof(int));
    recv_buffer = (int*)malloc(MESSAGE_SIZE * sizeof(int));

    if (!send_buffer || !recv_buffer) {
        printf("Memory allocation failed on process %d\n", rank);
        MPI_Finalize();
        return 1;
    }
    
    for(int i = 0; i < MESSAGE_SIZE; i++) send_buffer[i] = rank;

    printf("Process %d attempting standard send and receive with large message...\n", rank);

    MPI_Send(send_buffer, MESSAGE_SIZE, MPI_INT, 1 - rank, 0, MPI_COMM_WORLD);
    printf("Process %d finished sending (unlikely to print).\n", rank);

    MPI_Recv(recv_buffer, MESSAGE_SIZE, MPI_INT, 1 - rank, 0, MPI_COMM_WORLD, &status);
    printf("Process %d received data.\n", rank);
    
    free(send_buffer);
    free(recv_buffer);

    MPI_Finalize();
    return 0;
}
