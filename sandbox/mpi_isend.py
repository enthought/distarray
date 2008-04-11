from mpi4py import MPI
import numpy as np

size = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank
comm = MPI.COMM_WORLD

# data = np.arange(5,dtype=float)
data = np.array(100.0,dtype=float)
tag = 99
if rank==0:
    print "[0] Sending: ", data
    request = comm.Isend([data, MPI.FLOAT], 1, tag)
    request.Wait()
else:
    print "[1] Receiving..."
    comm.Recv([data, MPI.FLOAT], 0, tag)
    print "[1] Data: ", data
