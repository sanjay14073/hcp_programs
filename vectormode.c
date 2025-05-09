#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#define N 100
#define MAX_ITER 100

double phi[N+2][N+2], new_phi[N+2][N+2];

int main(int argc, char *argv[]) {
    int rank, size, i, j, iter;
    MPI_Request reqs[4];
    MPI_Status stats[4];
    int req_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize phi array (optional)
    for (i = 0; i < N + 2; i++)
        for (j = 0; j < N + 2; j++)
            phi[i][j] = rank;

    for (iter = 0; iter < MAX_ITER; iter++) {
        if (rank == 0)
            printf("Iteration %d starting on rank %d\n", iter, rank);

        // Exchange boundary rows with neighbors
        req_count = 0;
        if (rank > 0)
            MPI_Irecv(&phi[0][1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        if (rank < size - 1)
            MPI_Irecv(&phi[N+1][1], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        if (rank > 0)
            MPI_Isend(&phi[1][1], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        if (rank < size - 1)
            MPI_Isend(&phi[N][1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);

        // Compute inner region (excluding ghost rows)
        #pragma omp parallel for private(j)
        for (i = 2; i < N; i++) {
            for (j = 1; j <= N; j++) {
                new_phi[i][j] = 0.25 * (phi[i-1][j] + phi[i+1][j] + phi[i][j-1] + phi[i][j+1]);
            }
        }

        if (req_count > 0)
            MPI_Waitall(req_count, reqs, stats);

        // Compute boundary rows that depend on received data
        #pragma omp parallel for private(j)
        for (i = 1; i <= N; i += N - 1) {  // i = 1 and i = N
            for (j = 1; j <= N; j++) {
                new_phi[i][j] = 0.25 * (phi[i-1][j] + phi[i+1][j] + phi[i][j-1] + phi[i][j+1]);
            }
        }

        // Update phi from new_phi
        #pragma omp parallel for private(j)
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                phi[i][j] = new_phi[i][j];
            }
        }

        if (rank == 0)
            printf("Iteration %d finished on rank %d\n", iter, rank);
    }

    printf("Rank %d completed all iterations.\n", rank);

    MPI_Finalize();
    return 0;
}
