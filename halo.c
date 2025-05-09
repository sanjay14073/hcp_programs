#include <mpi.h>
#include <stdio.h>
#define XMAX 4
#define YMAX 4

void print_matrix(double mat[XMAX][YMAX], int rank, const char *msg) {
    printf("Process %d - %s:\n", rank, msg);
    for (int i = 0; i < XMAX; i++) {
        for (int j = 0; j < YMAX; j++) {
            printf("%5.1f ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    double matrix[XMAX][YMAX];
    MPI_Datatype rowtype;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize matrix
    for (int i = 0; i < XMAX; i++)
        for (int j = 0; j < YMAX; j++)
            matrix[i][j] = rank + 1.0;

    // Print initial state
    print_matrix(matrix, rank, "before communication");

    // Create derived datatype to send a column
    MPI_Type_vector(YMAX, 1, XMAX, MPI_DOUBLE, &rowtype);
    MPI_Type_commit(&rowtype);

    if (rank == 0) {
        MPI_Send(&matrix[2][0], 1, rowtype, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent column 2 to Process 1\n");
    } else if (rank == 1) {
        MPI_Recv(&matrix[0][0], 1, rowtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received column into column 0\n");
    }

    // Print final state
    print_matrix(matrix, rank, "after communication");

    MPI_Type_free(&rowtype);
    MPI_Finalize();
    return 0;
}
