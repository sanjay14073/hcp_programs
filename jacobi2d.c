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
    int req_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        printf("MPI Jacobi computation starting with %d processes...\n", size);

    for (iter = 0; iter < MAX_ITER; iter++) {
        req_count = 0;

        if (rank > 0)
            MPI_Irecv(&phi[0][1], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        if (rank < size - 1)
            MPI_Irecv(&phi[N+1][1], N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        if (rank > 0)
            MPI_Isend(&phi[1][1], N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        if (rank < size - 1)
            MPI_Isend(&phi[N][1], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);

        // Compute inner region
        for (i = 2; i < N; i++)
            for (j = 1; j <= N; j++)
                new_phi[i][j] = 0.25 * (phi[i-1][j] + phi[i+1][j] + phi[i][j-1] + phi[i][j+1]);

        // Wait only on the actual number of requests issued
        if (req_count > 0)
            MPI_Waitall(req_count, reqs, stats);

        // Update whole region
        for (i = 1; i <= N; i++)
            for (j = 1; j <= N; j++)
                phi[i][j] = new_phi[i][j];

        // Optional: print progress every 20 iterations
        if (rank == 0 && iter % 20 == 0)
            printf("Rank %d completed iteration %d\n", rank, iter);
    }

    // Final print
    if (rank == 0) {
        printf("Computation finished after %d iterations.\n", MAX_ITER);
        printf("Sample output: phi[50][50] = %.6f\n", phi[50][50]);

      
        
        printf("Final phi values (center 5x5 block):\n");
        for (i = 48; i <= 52; i++) {
            for (j = 48; j <= 52; j++)
                printf("%.2f ", phi[i][j]);
            printf("\n");
        }
        
    }

    MPI_Finalize();
    return 0;
}
