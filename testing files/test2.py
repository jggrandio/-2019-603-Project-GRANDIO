from mpi4py import MPI
import time



comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
    for i in range(5):
        comm.isend(i, dest=1, tag=12)
        time.sleep (0.6)

if rank == 1:
    req = comm.irecv(source=0, tag=12)
    for i in range(5):
        got,n = req.test()
        print(got)
        if got:
            req = comm.irecv(source=0, tag=12)
        #if got:
        print(n)
        time.sleep(0.5)