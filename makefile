CC = mpicc
CFLAGS = -O2 -fopenmp

TARGETS = jacobi_mpi jacobi_hybrid_task reduction_nonblocking halo_exchange jacobi_hybrid_vector

all: $(TARGETS)

jacobi_mpi: jacobi3d.c
	$(CC) $(CFLAGS) -o $@ $<

jacobi_hybrid_task: mpipuldhybrid.c
	$(CC) $(CFLAGS) -o $@ $<

reduction_nonblocking: manualmpi.c
	$(CC) $(CFLAGS) -o $@ $<

halo_exchange: halo.c
	$(CC) $(CFLAGS) -o $@ $<

jacobi_hybrid_vector: vectormode.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	mpirun -np 4 ./jacobi_mpi
	mpirun -np 4 ./jacobi_hybrid_task
	mpirun -np 4 ./reduction_nonblocking
	mpirun -np 2 ./halo_exchange
	mpirun -np 4 ./jacobi_hybrid_vector

clean:
	rm -f $(TARGETS)
