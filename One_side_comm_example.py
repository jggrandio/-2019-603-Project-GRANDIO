from mpi4py import MPI
import time
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main():
    buff = np.zeros(10, dtype='d')
    win = MPI.Win.Create( buff,comm=comm )

    if rank == 0:
        buff[:] = np.arange(10)
        win.Lock(rank)
        win.Put([buff, MPI.DOUBLE], target_rank=rank)
        win.Unlock(rank)
        buff =buff-0.5
        win.Lock(rank)
        win.Put([buff, MPI.DOUBLE], target_rank=rank)
        win.Unlock(rank)
    else:
        time.sleep(1)
        print('buff (rank1) before:', buff)
        win.Lock(0)
        win.Get([buff, MPI.DOUBLE], target_rank=0)
        win.Unlock(0)
        print('buff (rank1) after: ', buff)

    win.Free()

if __name__=='__main__':
    main()
