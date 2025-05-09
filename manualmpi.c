#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, i;
    double local_sum = 0.0, result = 0.0;
    double *recv_buffer;
    MPI_Request *requests;
    MPI_Status *statuses;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_sum = rank + 1.0;  // Each process contributes its rank + 1
    printf("Process %d has local_sum = %f\n", rank, local_sum);

    if (rank == 0) {
        recv_buffer = malloc((size - 1) * sizeof(double));
        requests = malloc((size - 1) * sizeof(MPI_Request));
        statuses = malloc((size - 1) * sizeof(MPI_Status));

        for (i = 1; i < size; i++) {
            MPI_Irecv(&recv_buffer[i - 1], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }

        MPI_Waitall(size - 1, requests, statuses);

        result = local_sum;
        for (i = 0; i < size - 1; i++) {
            printf("Received %f from process %d\n", recv_buffer[i], i + 1);
            result += recv_buffer[i];
        }

        printf("Total sum = %f\n", result);

        free(recv_buffer);
        free(requests);
        free(statuses);
    } else {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
