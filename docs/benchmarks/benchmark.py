import distarray as da
from distarray.mpi import (
    MPI, 
    create_comm_of_size,
    create_comm_with_list)

from distarray.core.error import *
from distarray.mpi.error import *

import numpy as np
import timeit

def benchmark_function(f, *args, **kwargs):
    comm_size = MPI.COMM_WORLD.Get_size()
    comm_rank = MPI.COMM_WORLD.Get_rank()
    
    if comm_rank==0:
        print
        print "Testing..."
        print f.__doc__
        print "args: ", args
        print "kwargs: ", kwargs
    size = 1
    while size <= comm_size:
        try:
            comm = create_comm_of_size(size)
        except InvalidCommSizeError:
            pass
        else:
            try:
                t1 = MPI.Wtime()
                f(comm, *args, **kwargs)
                t2 = MPI.Wtime()
                if comm_rank==0:
                    print "Time for size=%s on rank=%s: %s" % (size, comm_rank, t2-t1)
            except:
                pass
            else:
                "It failed"
            size *= 2