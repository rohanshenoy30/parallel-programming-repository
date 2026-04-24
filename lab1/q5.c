#include <stdio.h>
#include <mpi.h>

long long factorial(int n) {
    if (n == 0) return 1;
    long long res = 1;
    for (int i = 1; i <= n; i++) res *= i;
    return res;
}


long long fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    long long a = 0, b = 1, temp;
    for (int i = 2; i <= n; i++) {
        temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main(int argc, char** argv) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank % 2 == 0) {
        long long fact = factorial(rank);
        printf("Rank %d (Even): Factorial is %lld\n", rank, fact);
    } else {
        long long fib = fibonacci(rank);
        printf("Rank %d (Odd): Fibonacci number is %lld\n", rank, fib);
    }

    MPI_Finalize();
    return 0;
}
