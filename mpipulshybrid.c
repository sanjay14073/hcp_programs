#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#define N 100
#define MAX_ITER 100

double phi[N+2][N+2], new_phi[N+2][N+2];

int main(int argc, char *argv[]) {
    int rank, size, i, j, iter, tid;
    MPI_Request reqs[4];
    MPI_Status stats[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #pragma omp parallel private(tid, i, j)
    {
        tid = omp_get_thread_num();
        for (iter = 0; iter < MAX_ITER; iter++) {
            if (tid == 0) {
                if (rank > 0)
                    MPI_Irecv(&phi[0][1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[0]);
                if (rank < size - 1)
                    MPI_Irecv(&phi[N+1][1], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[1]);
                if (rank > 0)
                    MPI_Isend(&phi[1][1], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[2]);
                if (rank < size - 1)
                    MPI_Isend(&phi[N][1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[3]);
            } else {
                int chunk = (N - 2) / (omp_get_num_threads() - 1);
                int start = 2 + (tid - 1) * chunk;
                int end = start + chunk;
                if (end > N - 1) end = N - 1;
                for (i = start; i <= end; i++)
                    for (j = 1; j <= N; j++)
                        new_phi[i][j] = 0.25 * (phi[i-1][j] + phi[i+1][j] + phi[i][j-1] + phi[i][j+1]);
            }
            #pragma omp barrier
            if (tid == 0)
                MPI_Waitall(4, reqs, stats);
            #pragma omp barrier
        }
    }

    MPI_Finalize();
    return 0;
}
