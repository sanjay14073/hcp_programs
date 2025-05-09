#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#define N 100
#define MAX_ITER 100

double phi[N+2][N+2], new_phi[N+2][N+2];

int main(int argc, char *argv[]) {
    int rank, size, i, j, iter;
    MPI_Request reqs[4];
    MPI_Status stats[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (iter = 0; iter < MAX_ITER; iter++) {
        if (rank > 0)
            MPI_Irecv(&phi[0][1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
        if (rank < size - 1)
            MPI_Irecv(&phi[N+1][1], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[1]);
        if (rank > 0)
            MPI_Isend(&phi[1][1], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[2]);
        if (rank < size - 1)
            MPI_Isend(&phi[N][1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[3]);

        for (i = 2; i < N; i++)
            for (j = 1; j <= N; j++)
                new_phi[i][j] = 0.25 * (phi[i-1][j] + phi[i+1][j] + phi[i][j-1] + phi[i][j+1]);

        MPI_Waitall(4, reqs, stats);

        for (i = 1; i <= N; i++)
            for (j = 1; j <= N; j++)
                phi[i][j] = new_phi[i][j];
    }

    MPI_Finalize();
    return 0;
}
