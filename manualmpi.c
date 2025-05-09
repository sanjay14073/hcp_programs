#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, i;
    double local_sum = 0.0, recv_buffer, result = 0.0;
    MPI_Request *requests;
    MPI_Status *statuses;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_sum = rank + 1.0;
    if (rank == 0) {
        requests = malloc((size - 1) * sizeof(MPI_Request));
        statuses = malloc((size - 1) * sizeof(MPI_Status));
        for (i = 1; i < size; i++)
            MPI_Irecv(&recv_buffer, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        MPI_Waitall(size - 1, requests, statuses);
        result = local_sum;
        for (i = 1; i < size; i++)
            result += recv_buffer;
    } else {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
        printf("Total sum = %f\n", result);

    MPI_Finalize();
    return 0;
}
