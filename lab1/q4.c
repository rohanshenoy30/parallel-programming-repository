#include <stdio.h>
#include <string.h>
#include <ctype.h>  
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    char str[100] = "HELLO"; 
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    MPI_Bcast(str, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    int len = strlen(str);

    if (rank < len) {
        char original = str[rank];

        if (isupper(str[rank])) {
            str[rank] = tolower(str[rank]);
        } else if (islower(str[rank])) {
            str[rank] = toupper(str[rank]);
        }

        printf("Process %d toggled '%c' to '%c' -> Current state: %s\n", 
               rank, original, str[rank], str);
    }

    MPI_Finalize();
    return 0;
}
