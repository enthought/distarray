# encoding: utf-8

__docformat__ = "restructuredtext en"

#----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

from distarray.mpi import (
    MPI, 
    create_comm_of_size,
    create_comm_with_list)

from distarray.core.error import *
from distarray.mpi.error import *

import numpy as np

#----------------------------------------------------------------------------
# Exports
#----------------------------------------------------------------------------

__all__ = [
    'benchmark_function'
]


def benchmark_function(f, *args, **kwargs):
    comm_size = MPI.COMM_WORLD.Get_size()
    comm_rank = MPI.COMM_WORLD.Get_rank()
    
    sizes = []
    times = []
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
                times.append(t2-t1)
                sizes.append(size)
            except MPICommError:
                pass
            else:
                "It failed"
            size *= 2
    return (sizes, times)
        
        