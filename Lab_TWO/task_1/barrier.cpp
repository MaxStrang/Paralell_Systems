#include "barrier.hpp"

void barrier(MPI_Comm COMM_BARRIER) {
	int rank, size;
	MPI_Comm_rank(COMM_BARRIER, &rank);
	MPI_Comm_size(COMM_BARRIER, &size);

	//Forward signal
	if(rank != 0)
	{
		//All processes wait for ready signal from process(n-1)
		MPI_Recv(NULL, 0, MPI_BYTE, rank - 1, 0, COMM_BARRIER, MPI_STATUS_IGNORE);
	}
	//As long as the next process isn't the last one, we notify it that the process before is ready.
	if(rank != size -1)
	{
		MPI_Send(NULL, 0, MPI_BYTE, rank + 1, 0, COMM_BARRIER);
	}
	//Backward signal
	if(rank != size - 1)
	{
		MPI_Recv(NULL, 0, MPI_BYTE, rank + 1, 1, COMM_BARRIER, MPI_STATUS_IGNORE);
	}
	if(rank != 0)
	{
		MPI_Send(NULL, 0, MPI_BYTE, rank - 1, 1, COMM_BARRIER);
	}
}
