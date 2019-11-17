from mpi4py import MPI
import numpy as np
import time
import DQNagent


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

amode = MPI.MODE_RDWR|MPI.MODE_CREATE
comm = MPI.COMM_WORLD
fh = MPI.File.Open(comm, "./datafile.contig", amode)

agent = DQNagent.agent(5,5,gamma=0.999 , epsilon = 1.0, epsilon_min=0.001,epsilon_decay=0.992, learning_rate=0.001, batch_size=128)

buffer = agent.model.get_weights()

if rank == 0:
    print(buffer)
    fh.Write(buffer)
if rank == 1:

    time.sleep(1)
    fh.Read (buffer)
    print(buffer)

fh.Close()
